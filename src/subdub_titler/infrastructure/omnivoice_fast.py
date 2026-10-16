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
        at = getattr(gen_config, "text_guidance", None)
        as_ = getattr(gen_config, "reference_guidance", None)
        beta = getattr(gen_config, "accent_guidance_beta", 0.0)
        is_dual = (at is not None or as_ is not None) and (at != as_)
        is_aag = (beta > 0.0)
        # Con un solo elemento y CFG simétrico estándar sin AAG, la pasada conjunta original
        # lanza menos kernels. Con CFG dual (at != as) o AAG (beta > 0), lote 1 pasa por
        # _generate_split para aplicar las ramas auxiliares correspondientes.
        if task.batch_size == 1 and not is_dual and not is_aag:
            return original(task, gen_config)
        return _generate_split(self, task, gen_config)

    model._generate_iterative = types.MethodType(generate, model)
    return True


def aag_cfg_log_probs(
    u_logits,
    c_logits,
    t_logits,
    at: float,
    as_: float,
    proxy_src_logits: list | None = None,
    proxy_tgt_logits: list | None = None,
    beta: float = 0.0,
):
    """Fórmula de CFG con Accent Analogy Guidance (Cho & Lee, 2026):
    ℓ_AAG = ℓu + at · (ℓt − ℓu) + as · (ℓc − ℓt) − (β / M) · Σm [ℓproxy_src,m − ℓproxy_tgt,m]
    Normalizado con log_softmax sobre las distribuciones combinadas.
    """
    import torch
    import torch.nn.functional as F

    u_log = F.log_softmax(u_logits, dim=-1)
    c_log = F.log_softmax(c_logits, dim=-1)
    if t_logits is None or at == as_:
        # Cuando at == as_, la fórmula ℓu + as*(ℓc - ℓu) equivale a c + (as - 1)*(c - u).
        base_guided = c_log + (as_ - 1.0) * (c_log - u_log)
    else:
        t_log = F.log_softmax(t_logits, dim=-1)
        base_guided = u_log + at * (t_log - u_log) + as_ * (c_log - t_log)

    if beta > 0.0 and proxy_src_logits and proxy_tgt_logits:
        M = min(len(proxy_src_logits), len(proxy_tgt_logits))
        if M > 0:
            bias = torch.zeros_like(base_guided)
            for m in range(M):
                ps_log = F.log_softmax(proxy_src_logits[m], dim=-1)
                pt_log = F.log_softmax(proxy_tgt_logits[m], dim=-1)
                bias = bias + (ps_log - pt_log)
            base_guided = base_guided - (beta / float(M)) * bias

    return F.log_softmax(base_guided, dim=-1)


def dual_cfg_log_probs(u_logits, c_logits, t_logits, at: float, as_: float):
    return aag_cfg_log_probs(u_logits, c_logits, t_logits, at, as_)


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
    from omnivoice.models.omnivoice import _filter_top_k, _get_time_steps, _gumbel_sample

    batch, channels = task.batch_size, self.config.num_audio_codebook
    vocab, pad_id, device = self.config.audio_vocab_size, self.config.audio_mask_id, self.device

    at = getattr(gen_config, "text_guidance", None)
    as_ = getattr(gen_config, "reference_guidance", None)
    beta = getattr(gen_config, "accent_guidance_beta", 0.0)
    proxies = getattr(gen_config, "accent_proxies", None) or ()
    if at is None:
        at = 1.0 + float(gen_config.guidance_scale)
    if as_ is None:
        as_ = 1.0 + float(gen_config.guidance_scale)
    need_text_branch = (at != as_)
    need_aag = (beta > 0.0 and len(proxies) > 0)
    if beta > 0.0 and not proxies:
        raise ValueError(
            "Modo AAG activo (accent_guidance_beta > 0), pero no se suministraron proxies válidos"
        )

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

    if need_text_branch:
        text_inputs = [
            self._prepare_inference_inputs(task.texts[i], task.target_lens[i], None,
                                           None, task.langs[i], task.instructs[i],
                                           gen_config.denoise)
            for i in range(batch)
        ]
        text_lens = [item["input_ids"].size(2) for item in text_inputs]
        text_branch = _padded_batch(torch, text_inputs, text_lens, max(text_lens), device,
                                    pad_id, channels)
    else:
        text_inputs = None
        text_lens = None
        text_branch = None

    proxy_src_branches = []
    proxy_src_lens_list = []
    proxy_tgt_branches = []
    proxy_tgt_lens_list = []
    if need_aag:
        for p in proxies:
            s_in = [
                self._prepare_inference_inputs(
                    task.texts[i], task.target_lens[i], p.src_text,
                    p.src_audio_tokens, task.langs[i], task.instructs[i],
                    gen_config.denoise,
                )
                for i in range(batch)
            ]
            s_lens = [item["input_ids"].size(2) for item in s_in]
            proxy_src_branches.append(_padded_batch(
                torch, s_in, s_lens, max(s_lens), device, pad_id, channels
            ))
            proxy_src_lens_list.append(s_lens)

            t_in = [
                self._prepare_inference_inputs(
                    task.texts[i], task.target_lens[i], p.tgt_text,
                    p.tgt_audio_tokens, task.langs[i], task.instructs[i],
                    gen_config.denoise,
                )
                for i in range(batch)
            ]
            t_lens = [item["input_ids"].size(2) for item in t_in]
            proxy_tgt_branches.append(_padded_batch(
                torch, t_in, t_lens, max(t_lens), device, pad_id, channels
            ))
            proxy_tgt_lens_list.append(t_lens)

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
        text_states = hidden(*text_branch) if need_text_branch else None
        ps_states = [hidden(*b) for b in proxy_src_branches] if need_aag else None
        pt_states = [hidden(*b) for b in proxy_tgt_branches] if need_aag else None
        for i in range(batch):
            count = schedules[i][step]
            if count <= 0:
                continue
            c_len, t_len = cond_lens[i], target_lens[i]
            c_logits = logits(cond_states, i, c_len - t_len, t_len)
            u_logits = logits(uncond_states, i, 0, t_len)
            t_logits = (logits(text_states, i, text_lens[i] - t_len, t_len)
                        if need_text_branch else None)
            ps_logits = ([logits(ps_states[m], i, proxy_src_lens_list[m][i] - t_len, t_len)
                          for m in range(len(proxies))] if need_aag else None)
            pt_logits = ([logits(pt_states[m], i, proxy_tgt_lens_list[m][i] - t_len, t_len)
                          for m in range(len(proxies))] if need_aag else None)

            log_probs = aag_cfg_log_probs(
                u_logits, c_logits, t_logits, at, as_, ps_logits, pt_logits, beta
            )
            log_probs[..., pad_id] = -float("inf")
            if gen_config.class_temperature > 0.0:
                filtered_probs = _filter_top_k(log_probs, ratio=0.1)
                pred = _gumbel_sample(filtered_probs, gen_config.class_temperature).argmax(dim=-1)
            else:
                pred = log_probs.argmax(dim=-1)
            scores = log_probs.max(dim=-1)[0]
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
            if need_text_branch:
                text_branch[0][i:i + 1, :, text_lens[i] - t_len:text_lens[i]] = current
            if need_aag:
                for m in range(len(proxies)):
                    ps_l = proxy_src_lens_list[m][i]
                    pt_l = proxy_tgt_lens_list[m][i]
                    proxy_src_branches[m][0][i:i + 1, :, ps_l - t_len:ps_l] = current
                    proxy_tgt_branches[m][0][i:i + 1, :, pt_l - t_len:pt_l] = current
    return [tokens[i, :, :target_lens[i]] for i in range(batch)]
