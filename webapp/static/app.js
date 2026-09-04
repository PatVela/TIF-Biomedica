"use strict";

const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("file");
const btn = document.getElementById("btn");
const statusEl = document.getElementById("status");
const resultCard = document.getElementById("result");
const summaryBox = document.getElementById("summaryBox");
const plotImg = document.getElementById("plotImg");
const intervalList = document.getElementById("intervalList");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) {
    setStatus("Selecciona un archivo CSV.", "err");
    return;
  }
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);

  btn.disabled = true;
  setStatus("Clasificando…", "loading");
  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();
    if (!data.ok) {
      setStatus("Error: " + (data.error || "desconocido"), "err");
      return;
    }
    render(data);
    setStatus(
      `Hecho: ${data.n_samples} muestras → ${data.n_intervals} intervalos. ` +
      `Ritmo dominante: ${data.dominant}`,
      "ok"
    );
  } catch (err) {
    setStatus("Error de red: " + err.message, "err");
  } finally {
    btn.disabled = false;
  }
});

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = "status " + (cls || "");
}

function render(data) {
  resultCard.hidden = false;

  // summary chips
  summaryBox.innerHTML = "";
  (data.summary || []).forEach((s) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${s.label} · ${s.pct}% (${s.count} int.)`;
    chip.title = s.desc;
    summaryBox.appendChild(chip);
  });

  // plot
  plotImg.src = data.plot;

  // per-interval toggle list (cap to first 200 for readability)
  intervalList.innerHTML = "";
  const h = document.createElement("h3");
  h.textContent = `Intervalos (${data.n_intervals}) — clic para expandir`;
  intervalList.appendChild(h);
  const items = data.per_interval || [];
  const wrap = document.createElement("div");
  (items.length ? items : [])
    .slice(0, 200)
    .forEach((iv, i) => {
      const p = document.createElement("p");
      p.className = "iv";
      p.textContent = ` #${iv.idx} [${iv.label}] ${iv.desc} · prob ${iv.prob}`;
      wrap.appendChild(p);
    });
  intervalList.appendChild(wrap);
}
