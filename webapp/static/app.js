"use strict";

// ============================================================ i18n =========
const I18N = {
  es: {
    "app.title": "Clasificador de Electrocardiogramas",
    "app.subtitle": "Asistente automático de lectura de ECG (modelo Hannun et al., Nature Medicine 2019, datos PhysioNet).",
    "app.ready": "Modelo listo",
    "app.disclaimer": "⚠️ Herramienta de investigación. No sustituye la lectura de un cardiólogo ni debe usarse para diagnóstico clínico.",
    "app.disclaimer2": "Desarrollado como parte del curso Trabajo de Investigación · UNSA.",
    "app.summaryShort": "Asistente automático de lectura de ECG de una sola derivación (Hannun et al., Nature Medicine 2019 · PhysioNet).",
    "empty.title": "Aún no hay resultados",
    "empty.body": "Sube un ECG o usa un ejemplo rápido y pulsa «Analizar ECG». El resultado aparecerá aquí.",
    "quick.normal": "Ritmo normal", "quick.normalSub": "ejemplo · sin subir archivo",
    "quick.af": "Fibrilación auricular", "quick.afSub": "ejemplo · sin subir archivo",
    "quick.noise": "Ruido / artefacto", "quick.noiseSub": "ejemplo · sin subir archivo",
    "step1.quick": "Ejemplos rápidos (para demo)",
    "footer.about": "Sobre el proyecto",
    "footer.auth": "Ficha de autoría",
    "footer.author": "Autor/a",
    "footer.advisor": "Asesor/a",
    "footer.course": "Curso",
    "footer.version": "Versión",
    "footer.model": "Modelo",
    "footer.academic": "Sistema desarrollado con fines académicos de investigación.",
    "footer.refs": "Referencias",
    "footer.privacy": "Privacidad y seguridad",
    "footer.privacyText": "Los datos de paciente se usan solo en su navegador para el informe y se eliminan tras procesarse. Los archivos subidos no se conservan.",
    "footer.type": "Aplicación de investigación",
    "det.arch": "Arquitectura",
    "det.archVal": "CNN profunda (Hannun et al., Nature Medicine 2019) · 34 capas residuales / 1.3 M parámetros",
    "det.dataset": "Conjunto de datos",
    "det.datasetVal": "PhysioNet CinC2017 · ECG de una sola derivación · 300 Hz · segmentos de 256 muestras",
    "det.trained": "Entrenado",
    "det.model": "Ficha técnica del modelo",
    "det.artifacts": "Para incluir la matriz de confusión y el F1 por clase como imágenes, ejecuta el script de evaluación con --save_metrics_dir (ver README).",
    "confidence": "Confianza media del ritmo predominante",
    "methodology": "Enfoque: la señal se divide en segmentos de 256 muestras (a 300 Hz), se normaliza con la media/desviación global del conjunto de entrenamiento y se clasifica con una red neuronal convolucional profunda (réplica de Hannun et al., Nature Medicine 2019) entrenada sobre PhysioNet CinC2017. Cada segmento recibe una clase (normal, fibrilación auricular, otro ritmo, ruido) y se combinan para dar el ritmo predominante del registro. Esta vista muestra los artefactos de la evaluación para reproducibilidad.",
    "step1.title": "Cargar el electrocardiograma",
    "step1.intro": "Sube una grabación de ECG de una sola derivación (una línea del electrocardiograma) o usa una señal de prueba con un clic.",
    "step1.drop": "Arrastra el archivo aquí",
    "step1.dropSub": "o haz clic para seleccionarlo",
    "step1.formats": "Formatos:",
    "step1.help": "¿Qué formato debo usar?",
    "step1.helpCsv": "el primer número de la primera fila es la frecuencia de muestreo (Hz) y el resto son los valores de la señal.",
    "step1.helpOther": "Los archivos .mat, .dat y .npy de una sola derivación también se aceptan. Si la señal tiene otra frecuencia, se ajusta automáticamente.",
    "step1.name": "Paciente (opcional)",
    "step1.age": "Edad (opcional)",
    "step1.gt": "Diagnóstico conocido (opcional)",
    "step1.confirm": "Confirmo que la señal es un ECG de una sola derivación.",
    "btn.example": "Usar señal de prueba",
    "btn.analyze": "Analizar ECG",
    "btn.print": "Descargar informe (PDF)",
    "step2.title": "Resultado",
    "tab.vis": "Visualización",
    "tab.report": "Informe",
    "tab.detail": "Detalle técnico",
    "det.classes": "Ritmos reconocidos",
    "det.val": "Fe del entrenamiento (val)",
    "det.acc": "exactitud",
    "det.loss": "pérdida",
    "modal.title": "Análisis completado",
    "report.ready": "Informe listo para imprimir.",
    "file.ready": "Archivo listo",
    "file.of": "de",
    "ok": "Análisis completado.",
    "loading": "Analizando la señal…",
    "err.network": "Error de conexión: ",
    "err.analyze": "No se pudo analizar: ",
    "resampled": (a, b) => `La señal se ajustó automáticamente de ${Math.round(a)} Hz a ${Math.round(b)} Hz (la frecuencia que usa el modelo).`,
  },
  en: {
    "app.title": "Electrocardiogram Classifier",
    "app.subtitle": "Automated ECG reading assistant (Hannun et al., Nature Medicine 2019, PhysioNet data).",
    "app.ready": "Model ready",
    "app.disclaimer": "⚠️ Research tool. Does not replace a cardiologist's reading and must not be used for clinical diagnosis.",
    "app.disclaimer2": "Developed as part of the Research Work course · UNSA.",
    "app.summaryShort": "Automated single-lead ECG reading assistant (Hannun et al., Nature Medicine 2019 · PhysioNet).",
    "empty.title": "No results yet",
    "empty.body": "Upload an ECG or use a quick example and press “Analyze ECG”. The result will appear here.",
    "quick.normal": "Normal rhythm", "quick.normalSub": "example · no upload",
    "quick.af": "Atrial fibrillation", "quick.afSub": "example · no upload",
    "quick.noise": "Noise / artifact", "quick.noiseSub": "example · no upload",
    "step1.quick": "Quick examples (for demo)",
    "footer.about": "About the project",
    "footer.auth": "Authorship",
    "footer.author": "Author",
    "footer.advisor": "Advisor",
    "footer.course": "Course",
    "footer.version": "Version",
    "footer.model": "Model",
    "footer.academic": "System developed for academic research purposes.",
    "footer.refs": "References",
    "footer.privacy": "Privacy & security",
    "footer.privacyText": "Patient data is used only in your browser for the report and is removed after processing. Uploaded files are not kept.",
    "footer.type": "Research application",
    "det.arch": "Architecture",
    "det.archVal": "Deep CNN (Hannun et al., Nature Medicine 2019) · 34 residual layers / 1.3 M params",
    "det.dataset": "Dataset",
    "det.datasetVal": "PhysioNet CinC2017 · single-lead ECG · 300 Hz · 256-sample segments",
    "det.trained": "Trained",
    "det.model": "Model technical sheet",
    "det.artifacts": "To include the confusion matrix and per-class F1 as figures here, run the eval script with --save_metrics_dir (see README).",
    "confidence": "Mean confidence of the dominant rhythm",
    "methodology": "Approach: the signal is split into 256-sample segments (at 300 Hz), normalised with the global training mean/std, and classified by a deep convolutional neural network (replica of Hannun et al., Nature Medicine 2019) trained on PhysioNet CinC2017. Each segment gets a class (normal, atrial fibrillation, other, noise) and they are combined into the record's dominant rhythm. This view shows the evaluation artefacts for reproducibility.",
    "step1.title": "Upload the electrocardiogram",
    "step1.intro": "Upload a single-lead ECG recording (one line of the electrocardiogram) or use a test signal with one click.",
    "step1.drop": "Drag the file here",
    "step1.dropSub": "or click to select it",
    "step1.formats": "Formats:",
    "step1.help": "What format should I use?",
    "step1.helpCsv": "the first number of the first row is the sampling rate (Hz) and the rest are the signal values.",
    "step1.helpOther": "Single-lead .mat, .dat and .npy files are also accepted. If the signal has another rate, it is adjusted automatically.",
    "step1.name": "Patient (optional)",
    "step1.age": "Age (optional)",
    "step1.gt": "Known diagnosis (optional)",
    "step1.confirm": "I confirm the signal is a single-lead ECG.",
    "btn.example": "Use test signal",
    "btn.analyze": "Analyze ECG",
    "btn.print": "Download report (PDF)",
    "step2.title": "Result",
    "tab.vis": "Visualization",
    "tab.report": "Report",
    "tab.detail": "Technical detail",
    "det.classes": "Recognized rhythms",
    "det.val": "Training fit (val)",
    "det.acc": "accuracy",
    "det.loss": "loss",
    "modal.title": "Analysis complete",
    "report.ready": "Report ready to print.",
    "file.ready": "File ready",
    "file.of": "of",
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
const statusEl = $("status"), resultCard = $("result");
const diagnosisBox = $("diagnosisBox"), trafficBox = $("trafficBox"), summaryBox = $("summaryBox");
const gtBox = $("gtBox"), plotDiv = $("plotDiv"), intervalList = $("intervalList");
const trueLabelInput = $("trueLabel"), confirmCheck = $("confirmCheck"), fileCard = $("fileCard");
const emptyState = $("emptyState"), resultBody = $("resultBody");
let lastResult = null, currentFile = null;

// Institutional identity (from the header data-* attributes).
const instEl = document.querySelector(".inst-strip");
const INST_NAME = instEl ? instEl.dataset.institucion : "";
const instEscuela = instEl ? instEl.dataset.escuela : "";

function setStatus(msg, cls) { statusEl.textContent = msg; statusEl.className = "status " + (cls || ""); }
function setLoading(on) {
  btnClassify.disabled = on;
  btnClassify.classList.toggle("btn-loading", on);
  btnClassify.setAttribute("aria-busy", on ? "true" : "false");
  if (on) setStatus(t("loading"), "loading");
}

// Show the model name (from the checkpoint path) in the badge and footer.
function setModelName(info) {
  if (!info || !info.model_path) return;
  const rel = info.model_path.replace(/\\/g, "/").split("/").pop();
  const short = rel.split("-").pop();
  const label = short && short.length < 60 ? short : info.model_path;
  $("modelName").textContent = label;
  $("modelName").title = info.model_path;
  $("modelNameFooter").textContent = info.model_path;
}
function applyI18n() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  $("langEs").classList.toggle("active", LANG === "es");
  $("langEn").classList.toggle("active", LANG === "en");
  if (lastResult) render(lastResult);   // re-render dynamic content
  refreshFileCard();
}

// ============================================================ upload =======
async function postFile(fd) {
  setLoading(true);
  try {
    const resp = await fetch("/predict", { method: "POST", body: fd });
    const data = await resp.json();
    if (!data.ok) { setStatus(t("err.analyze") + (data.error || ""), "err"); return null; }
    render(data);
    setStatus(t("ok"), "ok");
    return data;
  } catch (err) { setStatus(t("err.network") + err.message, "err"); return null; }
  finally { setLoading(false); }
}

function refreshFileCard() {
  if (!currentFile) { fileCard.hidden = true; return; }
  fileCard.hidden = false;
  const size = currentFile.size > 1048576 ? (currentFile.size/1048576).toFixed(2)+" MB" : Math.round(currentFile.size/1024)+" KB";
  fileCard.innerHTML = `<span>📄</span><div><div class="fname">${currentFile.name}</div><div class="hint muted">${t("file.ready")} · ${size} · ${t("file.of")} ${new Date().toLocaleTimeString()}</div></div>`;
}

async function classify(file, name) {
  currentFile = file;
  refreshFileCard();
  const fd = new FormData();
  fd.append("file", file, name || file.name);
  if (trueLabelInput.value.trim()) fd.append("label", normalizeLabel(trueLabelInput.value.trim()));
  return await postFile(fd);
}
function normalizeLabel(s) {
  const map = { normal: "N", n: "N", af: "A", a: "A", "fibrilacion": "A", otro: "O", o: "O", ruido: "~", "~": "~", "|": "|" };
  const k = s.toLowerCase();
  return map[k] || s;
}
function handleFiles(files) {
  if (!files || !files.length) return;
  setStatus("Archivo: " + files[0].name, "loading");
  classify(files[0], files[0].name);
}

dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", e => { e.preventDefault(); dropzone.classList.remove("drag"); handleFiles(e.dataTransfer.files); });
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFiles(fileInput.files); });

btnClassify.addEventListener("click", () => {
  if (!confirmCheck.checked) { setStatus(t("step1.confirm"), "err"); statusEl.scrollIntoView({behavior:"smooth"}); return; }
  if (currentFile) { classify(currentFile, currentFile.name); }
  else if (fileInput.files.length) { handleFiles(fileInput.files); }
  else { setStatus("📄 Selecciona un archivo o usa la señal de prueba.", "err"); }
});

// Quick examples (Ritmo normal / AF / Ruido) -> /example with a `kind`.
async function runExample(kind) {
  const cards = document.querySelectorAll(".example-card");
  cards.forEach(c => c.disabled = true);
  setStatus(t("loading"), "loading");
  try {
    const resp = await fetch("/example", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind }),
    });
    const data = await resp.json();
    if (!data.ok) { setStatus(t("err.analyze") + data.error, "err"); return; }
    render(data); setStatus(t("ok"), "ok");
  } catch (err) { setStatus(t("err.network") + err.message, "err"); }
  finally { cards.forEach(c => c.disabled = false); }
}
document.querySelectorAll(".example-card[data-kind]").forEach(c =>
  c.addEventListener("click", () => runExample(c.dataset.kind)));

// ============================================================ render ======
function render(data) {
  lastResult = data;
  resultCard.hidden = false;
  emptyState.hidden = true;
  resultBody.hidden = false;
  setModelName(data.info);
  resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  renderNotice(data); renderDiagnosis(data); renderTraffic(data); renderSummary(data);
  renderGroundTruth(data); renderPlotly(data); renderIntervals(data); renderReport(data);
  selectTab("vis");
}
// Keep the empty state visible until a first result is shown.
emptyState.hidden = false;

function renderNotice(data) {
  const n = $("notice");
  if (data.resampled) { n.style.display = "block"; n.textContent = t("resampled", data.orig_fs, data.applied_fs); }
  else n.style.display = "none";
}
function dominantConfidence(data) {
  // Mean model confidence over the intervals classified as the dominant rhythm.
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
  diagnosisBox.innerHTML = `<div class="diagnosis-card" style="border-left-color:${cColor(lbl)}">
    <div class="diagnosis-name" style="color:${cColor(lbl)}">${info.name}</div>
    <div class="diagnosis-desc">${info.desc} — ${LANG==="es"?"fue el ritmo más frecuente":"it was the most frequent rhythm"} (${pct}% ${LANG==="es"?"de la señal":"of the signal"}).</div>
    ${conf !== null ? `<div class="diagnosis-conf">✦ ${t("confidence")}: <strong>${conf}%</strong></div>` : ""}
  </div>`;
}
function renderTraffic(data) {
  trafficBox.innerHTML = "";
  const order = ["N","A","O","~"];
  order.forEach(lbl => {
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
  summaryBox.innerHTML += `<p class="hint muted summary-hint">${LANG==="es"?"Cada color corresponde a un tramo de la señal. Pasa el cursor sobre la señal para ver el detalle.":"Each color matches a segment. Hover the signal for detail."}</p>`;
}
function renderGroundTruth(data) {
  gtBox.innerHTML = "";
  if (data.ground_truth) {
    const g = data.ground_truth;
    gtBox.innerHTML = `<div class="gt ${g.correct?"gt-ok":"gt-bad"}">${g.correct?"✔":"✘"} ${g.correct?(LANG==="es"?"Coincide con el diagnóstico indicado":"Matches the given diagnosis"):(LANG==="es"?"No coincide con el diagnóstico indicado":"Does not match the given diagnosis")} (${data.dominant})</div>`;
  }
}
// Clinical ECG grid: pinkish plot background + grid lines (paper ECG look).
function isDark() { return document.documentElement.dataset.theme === "dark"; }
function renderPlotly(data) {
  const p = data.plotly;
  if (!p || typeof Plotly === "undefined") { plotDiv.innerHTML = `<img class="plot" src="${data.plot}" alt="ECG">`; return; }
  const dark = isDark();
  const plotBg = dark ? "#1a1416" : "#fff5f5";
  const grid = dark ? "#3a2830" : "#f0bcbc";
  const layout = {
    title: LANG==="es"?"Señal de ECG con clasificación por tramos":"ECG signal with segment classification",
    xaxis: {
      title: LANG==="es"?"tiempo (s)":"time (s)",
      gridcolor: grid, showgrid: true, zeroline: false,
      dtick: 0.2, linecolor: grid,
    },
    yaxis: {
      title: LANG==="es"?"Amplitud (mV)":"Amplitude (mV)",
      gridcolor: grid, showgrid: true, zeroline: false,
      dtick: 0.5, linecolor: grid,
    },
    shapes: p.shapes, showlegend: false,
    margin: { l: 55, r: 15, t: 45, b: 45 }, hovermode: "x",
    paper_bgcolor: plotBg, plot_bgcolor: plotBg,
    font: { color: getComputedStyle(document.body).color },
  };
  Plotly.react(plotDiv, p.traces, layout, { responsive: true, displaylogo: false });
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
  const rows = (data.summary || []).map(s => `<tr><td>${cInfo(s.label).name}</td><td>${s.pct}%</td><td>${s.count}</td></tr>`).join("");
  const header = `
    <div class="report-head">
      <div class="r-inst">PROJECT_NAME_PLACEHOLDER</div>
      <div class="r-sub">${escapeHtml(INST_NAME)} · ${escapeHtml(instEscuela)}</div>
    </div>`.replace("PROJECT_NAME_PLACEHOLDER", t("app.title"));
  $("reportBody").innerHTML = header + `
    <h3>${LANG==="es"?"Informe ECG":"ECG Report"}</h3>
    <table>
      <tr><th>${LANG==="es"?"Paciente":"Patient"}</th><td>${escapeHtml(name)}</td></tr>
      <tr><th>${LANG==="es"?"Edad":"Age"}</th><td>${escapeHtml(age)}</td></tr>
      <tr><th>${LANG==="es"?"Fecha":"Date"}</th><td>${now}</td></tr>
      <tr><th>${LANG==="es"?"Ritmo predominante":"Dominant rhythm"}</th><td>${info.name} (${pct}%)</td></tr>
      <tr><th>${LANG==="es"?"Confianza":"Confidence"}</th><td>${conf !== null ? conf + "%" : "—"}</td></tr>
      <tr><th>${LANG==="es"?"Parámetros de registro":"Recording parameters"}</th><td>${fs} Hz · 1 derivación · duración ${(data.n_samples_in/fs).toFixed(1)} s</td></tr>
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
    <p class="hint muted" style="margin-top:.8rem">${t("app.disclaimer")}</p>`;
}

// ============================================================ tabs ========
function selectTab(name) {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === "panel-" + name));
  if (name === "rep" && typeof Plotly !== "undefined") Plotly.Plots.resize(plotDiv);
}
document.querySelectorAll(".tab").forEach(b => b.addEventListener("click", () => selectTab(b.dataset.tab)));
$("btnPrint").addEventListener("click", () => { selectTab("rep"); setTimeout(() => window.print(), 150); });

// ============================================================ misc ========
function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }

// theme
$( "themeToggle").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme;
  const nxt = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = nxt;
  try { localStorage.setItem("ecg-theme", nxt); } catch (e) {}
  if (lastResult) renderPlotly(lastResult);   // recolour ECG grid on theme change
});
try { const saved = localStorage.getItem("ecg-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch (e) {}

// language
$("langEs").addEventListener("click", () => { LANG = "es"; applyI18n(); });
$("langEn").addEventListener("click", () => { LANG = "en"; applyI18n(); });

// format help
const fmtHelp = $("fmtHelp"), fmtHelpBox = $("fmtHelpBox");
fmtHelp.addEventListener("click", () => fmtHelpBox.hidden = !fmtHelpBox.hidden);

// Show the validation-metrics figures only if the eval script exported them.
const metricsBox = $("metricsBox");
if (metricsBox) {
  fetch("/static/metrics/confusion_matrix.png", { method: "HEAD" })
    .then(r => { if (r.ok) metricsBox.hidden = false; })
    .catch(() => {});
}

// init
applyI18n();
