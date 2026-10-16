"""Página HTML autónoma (sin red) del panel de escucha: puntúa acento y parecido por voz, guarda
el avance en el navegador y al final muestra la media por variante y descarga las respuestas."""
import html
import json
from collections.abc import Sequence
from dataclasses import asdict

from subdub_titler.application.listening_panel import PanelItem

_PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panel de escucha</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --text:#1d1d1b; --muted:#6b6b66; --line:#e2e2dc;
        --accent:#2f6f5e; --chip:#eef3f1; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151614; --card:#1e201d; --text:#ecece8; --muted:#a2a39c; --line:#33352f;
          --accent:#7cc2ad; --chip:#243029; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:760px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; }
p.lead { color:var(--muted); margin:0 0 20px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:16px; margin-bottom:14px; }
.head { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
.num { color:var(--muted); font-size:13px; white-space:nowrap; }
.text { font-size:17px; margin:2px 0 10px; }
audio { width:100%; height:36px; }
.label { font-size:13px; color:var(--muted); margin:10px 0 2px; }
.voice { border-top:1px solid var(--line); margin-top:12px; padding-top:10px; }
.row { display:flex; align-items:center; gap:6px; margin-top:6px; }
.row span { font-size:13px; color:var(--muted); flex:0 0 100px; }
.row button { flex:1 1 0; max-width:48px; min-width:0; padding:6px 0;
              border:1px solid var(--line); border-radius:6px;
              background:var(--card); color:var(--text); font:inherit; cursor:pointer; }
.row button[aria-pressed=true] { background:var(--accent); border-color:var(--accent);
                                 color:var(--card); }
.bar { position:sticky; bottom:0; background:var(--bg); border-top:1px solid var(--line);
       padding:12px 16px; display:flex; gap:10px; align-items:center; justify-content:center;
       flex-wrap:wrap; }
.bar button { padding:8px 14px; border-radius:6px; border:1px solid var(--accent);
              background:var(--accent); color:var(--card); font:inherit; cursor:pointer; }
.bar button.ghost { background:transparent; color:var(--accent); }
table { width:100%; border-collapse:collapse; }
th, td { text-align:left; padding:6px 4px; border-bottom:1px solid var(--line); }
td.n { font-variant-numeric:tabular-nums; }
#results { display:none; }
</style>
</head>
<body>
<main>
<h1>Panel de escucha</h1>
<p class="lead">Escucha la voz original y después cada voz doblada. Puntúa el
<b>acento</b> (1 = claramente extranjero, 5 = nativo), el <b>parecido</b> con la voz original
(1 = otra persona, 5 = la misma) y la <b>naturalidad/expresividad</b> (1 = robótica o plana,
5 = humana y expresiva). El orden de las voces es aleatorio en cada frase.</p>
<section id="results" class="card"></section>
<div id="items"></div>
</main>
<div class="bar"><span id="progress" class="num"></span>
<button id="show">Ver resultados</button>
<button id="save" class="ghost">Descargar respuestas</button></div>
<script>
const PANEL = __PANEL__;
const KEY = "listen-panel:" + PANEL.id;
let ratings = {};
try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ratings = {}; }
const total = PANEL.items.reduce((n, item) => n + item.clips.length * 3, 0);

function persist() {
  try { localStorage.setItem(KEY, JSON.stringify(ratings)); } catch (e) {}
  const done = Object.values(ratings).reduce((n, r) => n + Object.keys(r).length, 0);
  document.getElementById("progress").textContent = done + " de " + total + " puntuaciones";
}

function scale(file, field, title) {
  const row = document.createElement("div");
  row.className = "row";
  const label = document.createElement("span");
  label.textContent = title;
  row.append(label);
  for (let v = 1; v <= 5; v++) {
    const b = document.createElement("button");
    b.textContent = v;
    b.setAttribute("aria-pressed", String((ratings[file] || {})[field] === v));
    b.onclick = () => {
      ratings[file] = Object.assign(ratings[file] || {}, {[field]: v});
      row.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      persist();
    };
    row.append(b);
  }
  return row;
}

function player(src) {
  const a = document.createElement("audio");
  a.controls = true; a.preload = "none"; a.src = src;
  return a;
}

const list = document.getElementById("items");
PANEL.items.forEach((item, i) => {
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = '<div class="head"><div class="text"></div><div class="num"></div></div>' +
                   '<div class="label">Voz original (inglés)</div>';
  card.querySelector(".text").textContent = item.text;
  card.querySelector(".num").textContent = (i + 1) + " / " + PANEL.items.length;
  card.append(player(item.original));
  item.clips.forEach((clip, j) => {
    const box = document.createElement("div");
    box.className = "voice";
    const label = document.createElement("div");
    label.className = "label";
    label.textContent = "Voz " + (j + 1);
    box.append(label, player(clip.file),
               scale(clip.file, "accent", "Acento (1–5)"),
               scale(clip.file, "similarity", "Parecido (1–5)"),
               scale(clip.file, "naturalness", "Naturalidad (1–5)"));
    card.append(box);
  });
  list.append(card);
});

function answers() {
  return PANEL.items.flatMap(item => item.clips.map(clip => Object.assign(
    {index: item.index, variant: clip.variant}, ratings[clip.file] || {})));
}

function summary() {
  const by = {};
  for (const a of answers()) {
    const s = by[a.variant] = by[a.variant] || {accent: [], similarity: [], naturalness: []};
    if (a.accent) s.accent.push(a.accent);
    if (a.similarity) s.similarity.push(a.similarity);
    if (a.naturalness) s.naturalness.push(a.naturalness);
  }
  const mean = xs => xs.length ? +(xs.reduce((x, y) => x + y, 0) / xs.length).toFixed(2) : null;
  return Object.fromEntries(Object.entries(by).map(([k, s]) => [k, {
    accent: mean(s.accent), similarity: mean(s.similarity),
    naturalness: mean(s.naturalness), rated: s.accent.length}]));
}

document.getElementById("show").onclick = () => {
  const box = document.getElementById("results");
  box.replaceChildren();
  const table = document.createElement("table");
  table.innerHTML = "<thead><tr><th>Variante</th><th>Acento</th><th>Parecido</th>" +
                    "<th>Naturalidad</th><th>Voces puntuadas</th></tr></thead>";
  const body = table.createTBody();
  for (const [name, s] of Object.entries(summary())) {
    const rowVals = [name, s.accent ?? "–", s.similarity ?? "–",
                     s.naturalness ?? "–", s.rated];
    rowVals.forEach((value, i) => {
      const cell = row.insertCell();
      cell.textContent = value;
      if (i) cell.className = "n";
    });
  }
  box.append(table);
  box.style.display = "block";
  window.scrollTo({top: 0, behavior: "smooth"});
};

document.getElementById("save").onclick = () => {
  const data = {panel: PANEL.id, title: PANEL.title, summary: summary(), answers: answers()};
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)],
                                           {type: "application/json"}));
  const a = Object.assign(document.createElement("a"), {href: url, download: "respuestas.json"});
  a.click();
  URL.revokeObjectURL(url);
};
persist();
</script>
</body>
</html>
"""


def render_listening_page(items: Sequence[PanelItem], title: str, panel_id: str) -> str:
    data = {"id": panel_id, "title": title, "items": [asdict(item) for item in items]}
    # `</` escapado: el JSON va dentro de <script>.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return _PAGE.replace("<title>Panel de escucha</title>",
                         f"<title>{html.escape(title)}</title>").replace("__PANEL__", payload)
