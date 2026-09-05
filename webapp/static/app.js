"use strict";

// ============================================================ i18n =========
const I18N = {
  es: {
    "app.title": "Clasificador de ECG",
    "app.subtitle": "Lectura automática de ECG de una sola derivación (Hannun et al., Nature Medicine 2019 · PhysioNet).",
    "app.ready": "Modelo listo",
    "app.disclaimer": "Herramienta de investigación — no reemplaza la lectura de un cardiólogo.",
    "app.disclaimer2": "Trabajo de Investigación · UNSA.",
    "nav.upload": "Cargar ECG",
    "nav.vis": "Resultado",
    "nav.detail": "Detalle Técnico",
    "step1.title": "Cargar electrocardiograma",
    "step1.intro": "Sube un ECG de una sola derivación o un ejemplo preconfigurado. El análisis inicia solo al pulsar «Analizar ECG».",
    "step1.drop": "Arrastra el archivo aquí",
    "step1.dropSub": "o haz clic para seleccionarlo",
    "step1.formats": "Formatos:",
    "step1.help": "¿Qué formato usar?",
    "step1.helpCsv": "el primer número de la primera fila es la frecuencia de muestreo (Hz) y el resto son los valores de la señal.",
    "step1.helpOther": "Los archivos .mat, .dat y .npy de una sola derivación también se aceptan. Si la señal tiene otra frecuencia, se ajusta automáticamente.",
    "step1.name": "Paciente (opcional)",
    "step1.age": "Edad (opcional)",
    "step1.gt": "Diagnóstico conocido (opcional)",
    "step1.quick": "Ejemplos rápidos",
    "quick.normal": "Ritmo normal", "quick.normalSub": "ejemplo · sin subir archivo",
    "quick.af": "Fibrilación auricular", "quick.afSub": "ejemplo · sin subir archivo",
    "quick.noise": "Ruido / artefacto", "quick.noiseSub": "ejemplo · sin subir archivo",
    "btn.analyze": "Analizar ECG",
    "btn.print": "Descargar informe (PDF)",
    "empty.title": "Aún no hay resultados",
    "empty.body": "Sube un ECG o usa un ejemplo y pulsa «Analizar ECG» para ver el resultado.",
    "leads.title": "Se detectaron múltiples derivaciones",
    "leads.text": "El sistema analizará una sola derivación. Selecciona cuál usar:",
    "vis.sub": "Trazado del ECG, clasificación por tramos y distribución.",
    "detail.sub": "Rigor metodológico: enfoque, ficha técnica y artefactos de validación.",
    "detail.empty": "Aún no hay detalle",
    "tab.vis": "Visualización",
    "tab.detail": "Detalle técnico",
    "det.classes": "Ritmos reconocidos",
    "det.val": "Fe del entrenamiento (val)",
    "det.acc": "exactitud", "det.loss": "pérdida",
    "det.arch": "Arquitectura",
    "det.archVal": "CNN profunda (Hannun et al., Nature Medicine 2019) · 34 capas residuales / 1.3 M parámetros",
    "det.dataset": "Conjunto de datos",
    "det.datasetVal": "PhysioNet CinC2017 · ECG de una sola derivación · 300 Hz · segmentos de 256 muestras",
    "det.trained": "Entrenado",
    "det.model": "Ficha técnica del modelo",
    "confidence": "Confianza media del ritmo predominante",
    "methodology": "Enfoque: la señal se divide en segmentos de 256 muestras (a 300 Hz), se normaliza con la media/desviación global del conjunto de entrenamiento y se clasifica con una red neuronal convolucional profunda (réplica de Hannun et al., Nature Medicine 2019) entrenada sobre PhysioNet CinC2017. Cada segmento recibe una clase (normal, fibrilación auricular, otro ritmo, ruido) y se combinan para dar el ritmo predominante del registro.",
    "footer.author": "Autor:", "footer.advisor": "Asesor:",
    "footer.model": "Modelo", "footer.refs": "Ref.:",
    "file.ready": "Archivo listo", "file.select": "Selecciona un archivo.",
    "ok": "Análisis completado.",
    "loading": "Analizando la señal…",
    "err.network": "Error de conexión: ",
    "err.analyze": "No se pudo analizar: ",
    "resampled": (a, b) => `La señal se ajustó automáticamente de ${Math.round(a)} Hz a ${Math.round(b)} Hz (la frecuencia que usa el modelo).`,
  },
  en: {
    "app.title": "ECG Classifier",
    "app.subtitle": "Automated single-lead ECG reading (Hannun et al., Nature Medicine 2019 · PhysioNet).",
    "app.ready": "Model ready",
    "app.disclaimer": "Research tool — does not replace a cardiologist's reading.",
    "app.disclaimer2": "Research Work course · UNSA.",
    "nav.upload": "Upload ECG",
    "nav.vis": "Result",
    "nav.detail": "Technical detail",
    "step1.title": "Upload electrocardiogram",
    "step1.intro": "Upload a single-lead ECG recording or a preset example. Analysis only starts when you press “Analyze ECG”.",
    "step1.drop": "Drag the file here",
    "step1.dropSub": "or click to select it",
    "step1.formats": "Formats:",
    "step1.help": "What format to use?",
    "step1.helpCsv": "the first number of the first row is the sampling rate (Hz) and the rest are the signal values.",
    "step1.helpOther": "Single-lead .mat, .dat and .npy files are also accepted. If the signal has another rate, it is adjusted automatically.",
    "step1.name": "Patient (optional)",
    "step1.age": "Age (optional)",
    "step1.gt": "Known diagnosis (optional)",
    "step1.quick": "Quick examples",
    "quick.normal": "Normal rhythm", "quick.normalSub": "example · no upload",
    "quick.af": "Atrial fibrillation", "quick.afSub": "example · no upload",
    "quick.noise": "Noise / artifact", "quick.noiseSub": "example · no upload",
    "btn.analyze": "Analyze ECG",
    "btn.print": "Download report (PDF)",
    "empty.title": "No results yet",
    "empty.body": "Upload an ECG or use an example and press “Analyze ECG” to see the result.",
    "leads.title": "Multiple leads detected",
    "leads.text": "The system will analyze a single lead. Choose which one to use:",
    "vis.sub": "ECG trace, segment classification and distribution.",
    "detail.sub": "Methodological rigor: approach, technical sheet and validation artefacts.",
    "detail.empty": "No detail yet",
    "tab.vis": "Visualization",
    "tab.detail": "Technical detail",
    "det.classes": "Recognized rhythms",
    "det.val": "Training fit (val)",
    "det.acc": "accuracy", "det.loss": "loss",
    "det.arch": "Architecture",
    "det.archVal": "Deep CNN (Hannun et al., Nature Medicine 2019) · 34 residual layers / 1.3 M params",
    "det.dataset": "Dataset",
    "det.datasetVal": "PhysioNet CinC2017 · single-lead ECG · 300 Hz · 256-sample segments",
    "det.trained": "Trained",
    "det.model": "Model technical sheet",
    "confidence": "Mean confidence of the dominant rhythm",
    "methodology": "Approach: the signal is split into 256-sample segments (at 300 Hz), normalised with the global training mean/std, and classified by a deep convolutional neural network (replica of Hannun et al., Nature Medicine 2019) trained on PhysioNet CinC2017. Each segment gets a class (normal, atrial fibrillation, other, noise) and they are combined into the record's dominant rhythm.",
    "footer.author": "Author:", "footer.advisor": "Advisor:",
    "footer.model": "Model", "footer.refs": "Ref.:",
    "file.ready": "File ready", "file.select": "Select a file.",
    "ok": "Analysis complete.",
    "loading": "Analyzing the signal…",
    "err.network": "Connection error: ",
    "err.analyze": "Could not analyze: ",
    "resampled": (a, b) => `The signal was auto-adjusted from ${Math.round(a)} Hz to ${Math.round(b)} Hz (the rate the model uses).`,
  },
};

const CLASS_INFO = {
  N: { es: { name: "Ritmo normal", desc: "Ritmo sinusal normal" }, en: { name: "Normal rhythm", desc: "Normal sinus rhythm" }, color: "#2ca02c" },
  A: { es: { name: "Fibrilación auricular", desc: "Fibrilación auricular / flutter" }, en: { name: "Atrial fibrillation", desc: "Atrial fibrillation / flutter" }, color: "#d62728" },
  O: { es: { name: "Otro ritmo", desc: "Otro ritmo cardíaco (no normal, no AF)" }, en: { name: "Other rhythm", desc: "Other cardiac rhythm (not normal, not AF)" }, color: "#ff7f0e" },
  "~": { es: { name: "Ruido / artefacto", desc: "Señal con ruido; lectura poco fiable" }, en: { name: "Noise / artifact", desc: "Noisy signal; unreliable reading" }, color: "#7f7f7f" },
  "|": { es: { name: "Sin clasificar", desc: "Silencio / sin clasificar" }, en: { name: "Unclassified", desc: "Silence / unclassified" }, color: "#1f77b4" },
};

let LANG = "es";
function t(key, ...args) {
  let v = (I18N[LANG][key] !== undefined) ? I18N[LANG][key] : key;
  if (typeof v === "function") return v(...args);
  return v;
}
function cInfo(lbl) { const c = CLASS_INFO[lbl] || {}; return (c[LANG] || c.es || {name: lbl, desc: lbl}); }
function cColor(lbl) { return (CLASS_INFO[lbl] || {}).color || "#9467bd"; }

// ============================================================ state =======
const $ = (id) => document.getElementById(id);
const dropzone = $("dropzone"), fileInput = $("file"), btnClassify = $("btnClassify");
const statusEl = $("status");
const diagnosisBox = $("diagnosisBox"), trafficBox = $("trafficBox"), summaryBox = $("summaryBox");
const gtBox = $("gtBox"), plotDiv = $("plotDiv"), intervalList = $("intervalList");
const trueLabelInput = $("trueLabel"), fileCard = $("fileCard");
const visContent = $("visContent"), visEmpty = $("visEmpty"),
      skel = $("skel"), reportBody = $("reportBody"),
      detContent = $("detContent"), detEmpty = $("detEmpty");
const leadsBanner = $("leadsBanner"), leadsSelect = $("leadsSelect"), leadsText = $("leadsText");

let lastResult = null, currentFile = null, pendingChannel = null;

const INST_ESCUELA = "UNSA · Escuela Profesional de Ingeniería Electrónica";

// ============================================================ utils =======
function setStatus(msg, cls) { statusEl.textContent = msg; statusEl.className = "status " + (cls || ""); }
function setLoading(on) {
  btnClassify.disabled = on;
  btnClassify.classList.toggle("btn-loading", on);
  btnClassify.setAttribute("aria-busy", on ? "true" : "false");
  if (on) skel.hidden = false;
}
function applyI18n() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  $("langEs").classList.toggle("active", LANG === "es");
  $("langEn").classList.toggle("active", LANG === "en");
  if (lastResult) render(lastResult);
  refreshFileCard();
  updateThemeIcon();
}
function setModelName(info) {
  if (!info || !info.model_path) return;
  const rel = info.model_path.replace(/\\/g, "/").split("/").pop();
  const short = rel.split("-").pop();
  const label = short && short.length < 60 ? short : info.model_path;
  $("modelName").textContent = label;
  $("modelName").title = info.model_path;
  $("modelNameFooter").textContent = info.model_path;
}
function updateThemeIcon() {
  const el = $("themeToggle");
  const dark = document.documentElement.dataset.theme === "dark";
  el.innerHTML = dark
    ? '<svg class="ico"><use href="#i-sun"/></svg>'
    : '<svg class="ico"><use href="#i-moon"/></svg>';
}

// ============================================================ sidebar nav =
function goTo(sec) {
  document.querySelectorAll(".nav-link").forEach(b => b.classList.toggle("active", b.dataset.sec === sec));
  ["upload", "vis", "det"].forEach(s => { const el = $("sec-" + s); if (el) el.hidden = (s !== sec); });
  window.scrollTo({ top: 0, behavior: "smooth" });
}
document.querySelectorAll(".nav-link").forEach(b => b.addEventListener("click", () => goTo(b.dataset.sec)));

// ============================================================ file upload =
function refreshFileCard() {
  if (!currentFile) { fileCard.hidden = true; return; }
  fileCard.hidden = false;
  const size = currentFile.size > 1048576 ? (currentFile.size/1048576).toFixed(2)+" MB" : Math.round(currentFile.size/1024)+" KB";
  fileCard.innerHTML = `<svg class="ico"><use href="#i-file"/></svg><div><div class="fname">${escapeHtml(currentFile.name)}</div><div class="hint muted">${size} · ${t("file.ready")}</div></div>`;
}

// NO auto-processing: only register the file and show the confirmation card.
function handleFiles(files) {
  if (!files || !files.length) return;
  currentFile = files[0];
  pendingChannel = null;
  leadsBanner.hidden = true;
  refreshFileCard();
  setStatus("", "");
}
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => { e.preventDefault(); dropzone.classList.remove("drag"); handleFiles(e.dataTransfer.files); });
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFiles(fileInput.files); });

function postFile(channel) {
  if (!currentFile) { setStatus(t("file.select"), "err"); return null; }
  skel.hidden = false; visEmpty.hidden = true; visContent.hidden = false;
  setLoading(true);
  const fd = new FormData();
  fd.append("file", currentFile, currentFile.name);
  if (channel !== undefined && channel !== null) fd.append("channel", channel);
  if (trueLabelInput.value.trim()) fd.append("label", normalizeLabel(trueLabelInput.value.trim()));
  return fetch("/predict", { method: "POST", body: fd })
    .then(r => r.json())
    .then(data => {
      if (!data.ok) { setStatus(t("err.analyze") + (data.error || ""), "err"); return null; }
      if (data.requires_channel) { showLeadsBanner(data); return "needs_channel"; }
      render(data);
      setStatus(t("ok"), "ok");
      return data;
    })
    .catch(err => { setStatus(t("err.network") + err.message, "err"); return null; })
    .finally(() => { skel.hidden = true; });
}
btnClassify.addEventListener("click", () => {
  if (!currentFile) { setStatus(t("file.select"), "err"); return; }
  const ch = pendingChannel !== null ? pendingChannel : undefined;
  postFile(ch);
});

// ============================================================ multi-lead ===
function showLeadsBanner(data) {
  pendingChannel = null;
  leadsBanner.hidden = false;
  leadsText.textContent = t("leads.text");
  leadsSelect.innerHTML = "";
  (data.channel_names || []).forEach((name, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = name;
    leadsSelect.appendChild(o);
  });
}
$("leadsContinue").addEventListener("click", () => {
  const ch = parseInt(leadsSelect.value, 10);
  if (isNaN(ch)) return;
  pendingChannel = ch;
  leadsBanner.hidden = true;
  postFile(ch);
});

// ============================================================ quick examples =
async function runExample(kind) {
  const cards = document.querySelectorAll(".example-card");
  cards.forEach(c => c.disabled = true);
  skel.hidden = false; visEmpty.hidden = true; visContent.hidden = false;
  setStatus(t("loading"), "loading");
  try {
    const resp = await fetch("/example", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind }) });
    const data = await resp.json();
    if (!data.ok) { setStatus(t("err.analyze") + data.error, "err"); return; }
    render(data); setStatus(t("ok"), "ok");
  } catch (err) { setStatus(t("err.network") + err.message, "err"); }
  finally { cards.forEach(c => c.disabled = false); skel.hidden = true; }
}
document.querySelectorAll(".example-card[data-kind]").forEach(c => c.addEventListener("click", () => runExample(c.dataset.kind)));

// ============================================================ render ======
function render(data) {
  lastResult = data;
  setModelName(data.info);

  visEmpty.hidden = true; visContent.hidden = false; skel.hidden = true;
  detContent.hidden = false; detEmpty.hidden = true;
  reportBody.hidden = true;

  renderNotice(data); renderDiagnosis(data); renderTraffic(data); renderSummary(data);
  renderGroundTruth(data); renderIntervals(data); renderReport(data);

  // Show the Result section BEFORE drawing the plot, so Plotly measures the
  // real (visible) container width and fills 100% of it.
  goTo("vis");
  renderPlotly(data);
}

function renderNotice(data) {
  const n = $("notice");
  if (data.resampled) { n.style.display = "block"; n.textContent = t("resampled", data.orig_fs, data.applied_fs); }
  else n.style.display = "none";
}
function dominantConfidence(data) {
  const lbl = data.dominant;
  const ivs = (data.per_interval || []).filter(iv => iv.label === lbl);
  if (!ivs.length) return null;
  const mean = ivs.reduce((a, iv) => a + (iv.prob || 0), 0) / ivs.length;
  return Math.round(mean * 1000) / 10;
}
function renderDiagnosis(data) {
  const lbl = data.dominant, info = cInfo(lbl);
  const pct = (data.summary && data.summary[0]) ? data.summary[0].pct : 0;
  const conf = dominantConfidence(data);
  const lead = data.channel_name ? ` · ${LANG==="es"?"derivación":"lead"} ${data.channel_name}` : "";
  diagnosisBox.innerHTML = `<div class="diagnosis-card" style="border-left-color:${cColor(lbl)}">
    <div class="diagnosis-name" style="color:${cColor(lbl)}">${info.name}</div>
    <div class="diagnosis-desc">${info.desc} — ${LANG==="es"?"fue el ritmo más frecuente":"it was the most frequent rhythm"} (${pct}% ${LANG==="es"?"de la señal":"of the signal"})${lead}.</div>
    ${conf !== null ? `<div class="diagnosis-conf">${t("confidence")}: <strong>${conf}%</strong></div>` : ""}
  </div>`;
}
function renderTraffic(data) {
  trafficBox.innerHTML = "";
  ["N","A","O","~"].forEach(lbl => {
    if (!data.summary.some(s => s.label === lbl)) return;
    const s = data.summary.find(x => x.label === lbl);
    const isDominant = lbl === data.dominant;
    const color = isDominant ? cColor(lbl) : (s.pct > 5 ? "#e0a800" : "#94a0b5");
    const label = isDominant ? (LANG==="es"?"predominante":"dominant") : (s.pct>5?(LANG==="es"?"presente":"present"):(LANG==="es"?"bajo":"low"));
    trafficBox.innerHTML += `<span class="traffic-item"><span class="traffic-dot" style="background:${color};border-color:${color}"></span>${cInfo(lbl).name} · ${label} (${s.pct}%)</span>`;
  });
}
function renderSummary(data) {
  summaryBox.innerHTML = `<div class="summary-label">${LANG==="es"?"Distribución del registro:":"Record distribution:"}</div>`;
  (data.summary || []).forEach(s => {
    const info = cInfo(s.label);
    summaryBox.innerHTML += `<span class="chip" style="background:${cColor(s.label)}" title="${info.desc} — ${s.count} ${LANG==="es"?"intervalos":"intervals"}">${info.name} · ${s.pct}%</span>`;
  });
  summaryBox.innerHTML += `<p class="hint muted summary-hint">${LANG==="es"?"Cada color corresponde a un tramo de la señal.":"Each color matches a segment."}</p>`;
}
function renderGroundTruth(data) {
  gtBox.innerHTML = "";
  if (data.ground_truth) {
    const g = data.ground_truth;
    gtBox.innerHTML = `<div class="gt ${g.correct?"gt-ok":"gt-bad"}">${g.correct?"✔":"✘"} ${g.correct?(LANG==="es"?"Coincide con el diagnóstico indicado":"Matches the given diagnosis"):(LANG==="es"?"No coincide con el diagnóstico indicado":"Does not match the given diagnosis")} (${data.dominant})</div>`;
  }
}
function isDark() { return document.documentElement.dataset.theme === "dark"; }
function renderPlotly(data) {
  const p = data.plotly;
  if (!p || typeof Plotly === "undefined") { plotDiv.innerHTML = `<img class="plot" src="${data.plot}" alt="ECG">`; return; }
  const dark = isDark();
  // Proper clinical "millimetre-paper" look: white/warm bg, a fine minor grid
  // (1mm = 0.04s / 0.1mV) and a slightly stronger major grid (5mm = 0.2s / 0.5mV).
  const bg = dark ? "#12151b" : "#fdfdfb";
  const gridMinor = dark ? "#2a3543" : "#f3dedd";
  const gridMajor = dark ? "#3c4a5c" : "#e6bab8";
  const gridEdge = dark ? "#33405a" : "#c9d4e2";
  const tickColor = dark ? "#c6d0e0" : "#35465c";
  const traceColor = "#111827";
  const layout = {
    title: { text: LANG==="es"?"Señal de ECG con clasificación por tramos":"ECG signal with segment classification", font: { size: 15, color: getComputedStyle(document.body).color }, x: 0.01 },
    autosize: true,
    xaxis: {
      title: LANG==="es"?"tiempo (s)":"time (s)",
      gridcolor: gridMajor, gridwidth: 1, showgrid: true, zeroline: false,
      showline: true, linecolor: gridEdge, mirror: true,
      dtick: 1, tickfont: { size: 11, color: tickColor }, titlefont: { size: 12 },
      automargin: true,
      minor: { showgrid: true, dtick: 0.2, gridcolor: gridMinor, gridwidth: 1 },
    },
    yaxis: {
      title: LANG==="es"?"Amplitud (mV)":"Amplitude (mV)",
      gridcolor: gridMajor, gridwidth: 1, showgrid: true, zeroline: false,
      showline: true, linecolor: gridEdge, mirror: true,
      dtick: 0.5, tickfont: { size: 11, color: tickColor }, titlefont: { size: 12 },
      automargin: true,
      minor: { showgrid: true, dtick: 0.1, gridcolor: gridMinor, gridwidth: 1 },
    },
    shapes: p.shapes, showlegend: false,
    margin: { l: 58, r: 20, t: 52, b: 48 }, hovermode: "x",
    paper_bgcolor: bg, plot_bgcolor: bg,
    font: { color: getComputedStyle(document.body).color },
  };
  const traces = p.traces.map(tr => ({ ...tr, line: { ...(tr.line||{}), color: traceColor, width: 1 } }));
  Plotly.react(plotDiv, traces, layout, { responsive: true, displaylogo: false });
  requestAnimationFrame(() => { try { Plotly.Plots.resize(plotDiv); } catch (e) {} });
}
function renderIntervals(data) {
  intervalList.innerHTML = "";
  (data.per_interval || []).forEach(iv => {
    intervalList.innerHTML += `<p class="iv"> #${iv.idx} · ${cInfo(iv.label).name} · ${LANG==="es"?"confianza":"confidence"} ${Math.round(iv.prob*100)}% · t≈${iv.start_ms} ms</p>`;
  });
}
function renderReport(data) {
  const info = cInfo(data.dominant);
  const pct = (data.summary && data.summary[0]) ? data.summary[0].pct : 0;
  const name = $("patName").value || "—", age = $("patAge").value || "—";
  const now = new Date().toLocaleString();
  const fs = Math.round(data.applied_fs || data.orig_fs || 300);
  const conf = dominantConfidence(data);
  const lead = data.channel_name || "";
  const rows = (data.summary || []).map(s => `<tr><td>${cInfo(s.label).name}</td><td>${s.pct}%</td><td>${s.count}</td></tr>`).join("");
  $("reportBody").innerHTML = `
    <div class="report-head">
      <div class="r-inst">${escapeHtml(t("app.title"))} — Informe ECG</div>
      <div class="r-sub">${escapeHtml(INST_ESCUELA)} · ${escapeHtml(t("app.disclaimer2"))}</div>
    </div>
    <table>
      <tr><th>${LANG==="es"?"Paciente":"Patient"}</th><td>${escapeHtml(name)}</td></tr>
      <tr><th>${LANG==="es"?"Edad":"Age"}</th><td>${escapeHtml(age)}</td></tr>
      <tr><th>${LANG==="es"?"Fecha":"Date"}</th><td>${now}</td></tr>
      <tr><th>${LANG==="es"?"Ritmo predominante":"Dominant rhythm"}</th><td>${info.name} (${pct}%)</td></tr>
      <tr><th>${LANG==="es"?"Confianza":"Confidence"}</th><td>${conf !== null ? conf + "%" : "—"}</td></tr>
      <tr><th>${LANG==="es"?"Parámetros de registro":"Recording parameters"}</th><td>${fs} Hz · 1 derivación${lead ? " ("+lead+")" : ""} · duración ${(data.n_samples_in/fs).toFixed(1)} s</td></tr>
    </table>
    <table>
      <tr><th>${LANG==="es"?"Clase":"Class"}</th><th>%</th><th>${LANG==="es"?"Intervalos":"Intervals"}</th></tr>
      ${rows}
    </table>
    <img src="${data.plot}" alt="ECG clasificado">
    <div class="r-sign">
      <div><div class="sig-line">${LANG==="es"?"Investigador/a":"Evaluator"}</div></div>
      <div><div class="sig-line">${LANG==="es"?"Asesor / validador":"Advisor / validator"}</div></div>
    </div>
    <p class="hint muted" style="margin-top:.9rem;color:#6b7a8d">${t("app.disclaimer")}</p>`;
}

// ============================================================ misc ========
function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function normalizeLabel(s) {
  const map = { normal: "N", n: "N", af: "A", a: "A", "fibrilacion": "A", otro: "O", o: "O", ruido: "~", "~": "~", "|": "|" };
  const k = s.toLowerCase();
  return map[k] || s;
}

// theme
try { const saved = localStorage.getItem("ecg-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch (e) {}
$("themeToggle").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme;
  const nxt = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = nxt;
  try { localStorage.setItem("ecg-theme", nxt); } catch (e) {}
  updateThemeIcon();
  if (lastResult) renderPlotly(lastResult);
});

// language
$("langEs").addEventListener("click", () => { LANG = "es"; applyI18n(); });
$("langEn").addEventListener("click", () => { LANG = "en"; applyI18n(); });

// format help
const fmtHelp = $("fmtHelp"), fmtHelpBox = $("fmtHelpBox");
fmtHelp.addEventListener("click", () => fmtHelpBox.hidden = !fmtHelpBox.hidden);

// Download a real, well-formatted PDF report (generated server-side).
async function downloadReport() {
  if (!lastResult) { setStatus(t("empty.body"), "err"); return; }
  const btn = $("btnPrint");
  btn.disabled = true;
  try {
    const b64 = (lastResult.plot || "").split(",")[1] || "";
    const pct = (lastResult.summary && lastResult.summary[0]) ? lastResult.summary[0].pct : null;
    const conf = dominantConfidence(lastResult);
    const info = cInfo(lastResult.dominant);
    const payload = {
      plot: b64,
      patient: $("patName").value || "",
      age: $("patAge").value || "",
      date: new Date().toLocaleString(),
      dominant: lastResult.dominant,
      dominant_name: info.name,
      dominant_pct: pct,
      confidence: conf,
      fs: Math.round(lastResult.applied_fs || lastResult.orig_fs || 300),
      duration: (lastResult.n_samples_in || 0) / (lastResult.applied_fs || 300),
      lead: lastResult.channel_name || "",
      classes: (lastResult.summary || []).map(s => ({ name: cInfo(s.label).name, pct: s.pct, count: s.count })),
    };
    const resp = await fetch("/report.pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) { const e = await resp.json().catch(()=>({})); throw new Error(e.error || resp.statusText); }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "informe_ecg.pdf";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    setStatus("Informe PDF descargado.", "ok");
  } catch (err) {
    setStatus(t("err.analyze") + (err.message || ""), "err");
  } finally { btn.disabled = false; }
}
$("btnPrint").addEventListener("click", downloadReport);

// show validation-metrics figures only if the eval script exported them
const metricsBox = $("metricsBox");
if (metricsBox) {
  fetch("/static/metrics/confusion_matrix.png", { method: "HEAD" })
    .then(r => { if (r.ok) metricsBox.hidden = false; }).catch(() => {});
}

// init
applyI18n();
