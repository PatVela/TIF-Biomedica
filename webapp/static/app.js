"use strict";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file");
const btnClassify = document.getElementById("btnClassify");
const btnExample = document.getElementById("btnExample");
const btnSwitch = document.getElementById("btnSwitch");
const modelSel = document.getElementById("modelSel");
const statusEl = document.getElementById("status");
const resultCard = document.getElementById("result");
const summaryBox = document.getElementById("summaryBox");
const gtBox = document.getElementById("gtBox");
const plotDiv = document.getElementById("plotDiv");
const intervalList = document.getElementById("intervalList");
const trueLabelInput = document.getElementById("trueLabel");

let lastResult = null;

// ------------------------------------------------------------------ helpers
function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = "status " + (cls || "");
}

function setLoading(on) {
  btnClassify.disabled = on;
  if (on) { setStatus("Clasificando…", "loading"); }
}

async function postFile(fd) {
  setLoading(true);
  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();
    if (!data.ok) { setStatus("Error: " + (data.error || "desconocido"), "err"); return null; }
    render(data);
    setStatus("Hecho.", "ok");
    return data;
  } catch (err) {
    setStatus("Error de red: " + err.message, "err");
    return null;
  } finally {
    setLoading(false);
  }
}

// ------------------------------------------------------------------ UI bits
function renderNotice(data) {
  const n = document.getElementById("notice");
  if (data.resampled) {
    n.style.display = "block";
    n.textContent = `Señal re-muestreada de ${Math.round(data.orig_fs)} Hz → ${Math.round(data.applied_fs)} Hz (frecuencia del modelo).`;
  } else {
    n.style.display = "none";
  }
}

function renderSummary(data) {
  summaryBox.innerHTML = "";
  (data.summary || []).forEach((s) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${s.label} · ${s.pct}% (${s.count} int.)`;
    chip.title = s.desc;
    summaryBox.appendChild(chip);
  });
}

function renderGroundTruth(data) {
  gtBox.innerHTML = "";
  if (data.ground_truth) {
    const g = data.ground_truth;
    const el = document.createElement("div");
    el.className = "gt " + (g.correct ? "gt-ok" : "gt-bad");
    el.textContent = (g.correct ? "✔" : "✘") +
      ` Coincide con la etiqueta real '${g.label}'` +
      (g.correct ? "" : ` (predicción: '${data.dominant}')`);
    gtBox.appendChild(el);
  }
}

function renderIntervals(data) {
  intervalList.innerHTML = "";
  const h = document.createElement("h3");
  h.textContent = `Intervalos (${data.n_intervals}) — clic para expandir`;
  intervalList.appendChild(h);
  const wrap = document.createElement("div");
  (data.per_interval || []).forEach((iv) => {
    const p = document.createElement("p");
    p.className = "iv";
    p.textContent = ` #${iv.idx} [${iv.label}] ${iv.desc} · prob ${iv.prob} · t≈${iv.start_ms} ms`;
    wrap.appendChild(p);
  });
  intervalList.appendChild(wrap);
}

// ------------------------------------------------------------------ Plotly
function renderPlotly(data) {
  const p = data.plotly;
  if (!p || typeof Plotly === "undefined") {
    // fallback to the static PNG if Plotly is unavailable
    plotDiv.innerHTML = `<img class="plot" src="${data.plot}" alt="ECG con predicción">`;
    return;
  }
  const layout = {
    title: "ECG — 1 predicción por intervalo de 256 muestras (sombreado = clase)",
    xaxis: { title: "tiempo (s)", gridcolor: "#eef1f6" },
    yaxis: { title: "Amplitud (mV)", gridcolor: "#eef1f6" },
    shapes: p.shapes,
    showlegend: false,
    margin: { l: 55, r: 15, t: 45, b: 45 },
    hovermode: "x",
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
  };
  // class legend as colored y-offset markers is overkill; rely on tooltip
  Plotly.react(plotDiv, p.traces, layout, { responsive: true, displaylogo: false });
}

// ------------------------------------------------------------------ render
function render(data) {
  lastResult = data;
  resultCard.hidden = false;
  renderNotice(data);
  renderSummary(data);
  renderGroundTruth(data);
  renderPlotly(data);
  renderIntervals(data);
  // prune the giant plot/images from the exported JSON
  data.__json = void 0;
}

// ------------------------------------------------------------------ upload
async function classify(file, name) {
  const fd = new FormData();
  fd.append("file", file, name || file.name);
  if (trueLabelInput.value.trim()) fd.append("label", trueLabelInput.value.trim());
  return await postFile(fd);
}

function handleFiles(files) {
  if (!files || !files.length) return;
  setStatus("Archivo: " + files[0].name, "loading");
  classify(files[0], files[0].name);
}

dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFiles(fileInput.files); });

// ------------------------------------------------------------------ example
btnExample.addEventListener("click", async () => {
  btnExample.disabled = true;
  setStatus("Generando señal de ejemplo…", "loading");
  try {
    const resp = await fetch("/example", { method: "POST" });
    const data = await resp.json();
    if (!data.ok) { setStatus("Error: " + data.error, "err"); return; }
    render(data);
    setStatus("Ejemplo generado y clasificado.", "ok");
  } catch (err) {
    setStatus("Error de red: " + err.message, "err");
  } finally {
    btnExample.disabled = false;
  }
});

// ------------------------------------------------------------------ model switch
btnSwitch.addEventListener("click", async () => {
  const rel = modelSel.value;
  if (!rel) return;
  btnSwitch.disabled = true;
  setStatus("Cargando modelo…", "loading");
  try {
    const resp = await fetch("/use_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: rel }),
    });
    const data = await resp.json();
    if (!data.ok) { setStatus("Error: " + data.error, "err"); return; }
    location.reload(); // refresh model info in the header
  } catch (err) {
    setStatus("Error de red: " + err.message, "err");
  } finally {
    btnSwitch.disabled = false;
  }
});

// ------------------------------------------------------------------ exports
function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById("btnExportJson").addEventListener("click", () => {
  if (!lastResult) return;
  const clean = { ...lastResult };
  delete clean.plot; delete clean.plotly; delete clean.__json;
  download("resultados.json", JSON.stringify(clean, null, 2));
});

document.getElementById("btnExportCsv").addEventListener("click", () => {
  if (!lastResult) return;
  const rows = [["idx", "label", "desc", "prob", "start_ms"]];
  (lastResult.per_interval || []).forEach((iv) =>
    rows.push([iv.idx, iv.label, iv.desc, iv.prob, iv.start_ms]));
  download("predicciones.csv", rows.map(r => r.join(",")).join("\n"));
});

document.getElementById("btnExportPng").addEventListener("click", () => {
  if (!lastResult) return;
  const a = document.createElement("a");
  a.href = lastResult.plot;
  a.download = "ecg_prediccion.png";
  a.click();
});
