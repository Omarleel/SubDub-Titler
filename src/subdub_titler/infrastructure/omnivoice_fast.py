"""Decodificación de OmniVoice con la rama incondicional de CFG en su propia pasada.

OmniVoice 0.2.1 apila condicional e incondicional en un único lote rellenado a la longitud
condicional (referencia + texto + objetivo, ~350 posiciones), aunque la incondicional solo
usa las del objetivo (~75). Separarlas evita calcular ese relleno: ×1.7 con lotes de 8 en la
RTX 5060 Ti. Mismo algoritmo: en fp32, 7/8 frases salen idénticas bit a bit (la restante
difiere por un empate numérico en el desenmascarado).

Acoplado a `omnivoice==0.2.1` (`_generate_iterative`, `_prepare_inference_inputs`); con otra
versión `install()` no hace nada y se usa la implementación original.
"""
import importlib.metadata
import logging
import math
import types

logger = logging.getLogger(__name__)
SUPPORTED_VERSION = "0.2.1"


def supported() -> bool:
    try:
        return importlib.metadata.version("omnivoice") == SUPPORTED_VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def install(model) -> bool:
    """Sustituye el decodificador solo en esta instancia (no parchea la clase)."""
    if not supported():
        logger.warning("omnivoice != %s: se usa la decodificación original", SUPPORTED_VERSION)
        return False
    original = model._generate_iterative

    def generate(self, task, gen_config):
        # Con un solo elemento, una pasada conjunta lanza menos kernels y es más rápida.
        if task.batch_size == 1:
            return original(task, gen_config)
        return _generate_split(self, task, gen_config)

    model._generate_iterative = types.MethodType(generate, model)
    return True


def _padded_batch(torch, inputs, lengths, width, device, pad_id, channels):
    batch = len(inputs)
    ids = torch.full((batch, channels, width), pad_id, dtype=torch.long, device=device)
    audio_mask = torch.zeros((batch, width), dtype=torch.bool, device=device)
    attention = torch.zeros((batch, 1, width, width), dtype=torch.bool, device=device)
    for i, item in enumerate(inputs):
        n = lengths[i]
        ids[i, :, :n] = item["input_ids"][0][..., -n:]
        audio_mask[i, :n] = item["audio_mask"][0][..., -n:]
        attention[i, :, :n, :n] = True
        if width > n:  # el relleno se atiende a sí mismo para evitar filas vacías en SDPA
            diagonal = torch.arange(n, width, device=device)
            attention[i, :, diagonal, diagonal] = True
    return ids, audio_mask, attention


def _schedules(time_steps, lengths, channels, num_step):
    schedules = []
    for length in lengths:
        total = remaining = length * channels
        schedule = []
        for step in range(num_step):
            count = remaining if step == num_step - 1 else min(
                math.ceil(total * (time_steps[step + 1] - time_steps[step])), remaining)
            schedule.append(int(count))
            remaining -= int(count)
        schedules.append(schedule)
    return schedules


def _generate_split(self, task, gen_config):
    import torch
    from omnivoice.models.omnivoice import _get_time_steps, _gumbel_sample

    batch, channels = task.batch_size, self.config.num_audio_codebook
    vocab, pad_id, device = self.config.audio_vocab_size, self.config.audio_mask_id, self.device
    inputs = [
        self._prepare_inference_inputs(task.texts[i], task.target_lens[i], task.ref_texts[i],
                                       task.ref_audio_tokens[i], task.langs[i],
                                       task.instructs[i], gen_config.denoise)
        for i in range(batch)
    ]
    cond_lens = [item["input_ids"].size(2) for item in inputs]
    target_lens = task.target_lens
    cond = _padded_batch(torch, inputs, cond_lens, max(cond_lens), device, pad_id, channels)
    uncond = _padded_batch(torch, inputs, target_lens, max(target_lens), device, pad_id,
                           channels)
    tokens = torch.full((batch, channels, max(target_lens)), pad_id, dtype=torch.long,
                        device=device)
    time_steps = _get_time_steps(t_start=0.0, t_end=1.0, num_step=gen_config.num_step,
                                 t_shift=gen_config.t_shift).tolist()
    schedules = _schedules(time_steps, target_lens, channels, gen_config.num_step)
    layer_ids = torch.arange(channels, device=device).view(1, -1, 1)

    def hidden(ids, audio_mask, attention):
        embeds = self._prepare_embed_inputs(ids, audio_mask)
        return self.llm(inputs_embeds=embeds, attention_mask=attention, return_dict=True)[0]

    def logits(states, row, start, length):
        # La cabeza de audio solo sobre las posiciones objetivo (no sobre referencia/texto).
        flat = self.audio_heads(states[row:row + 1, start:start + length])
        return flat.view(1, length, channels, vocab).permute(0, 2, 1, 3).float()

    for step in range(gen_config.num_step):
        cond_states = hidden(*cond)
        uncond_states = hidden(*uncond)
        for i in range(batch):
            count = schedules[i][step]
            if count <= 0:
                continue
            c_len, t_len = cond_lens[i], target_lens[i]
            pred, scores = self._predict_tokens_with_scoring(
                logits(cond_states, i, c_len - t_len, t_len),
                logits(uncond_states, i, 0, t_len), gen_config)
            scores = scores - layer_ids * gen_config.layer_penalty_factor
            if gen_config.position_temperature > 0.0:
                scores = _gumbel_sample(scores, gen_config.position_temperature)
            current = tokens[i:i + 1, :, :t_len]
            scores.masked_fill_(current != pad_id, -float("inf"))
            _, chosen = torch.topk(scores.flatten(), count)
            flat = current.flatten()
            flat[chosen] = pred.flatten()[chosen]
            current.copy_(flat.view_as(current))
            cond[0][i:i + 1, :, c_len - t_len:c_len] = current
            uncond[0][i:i + 1, :, :t_len] = current
    return [tokens[i, :, :target_lens[i]] for i in range(batch)]
