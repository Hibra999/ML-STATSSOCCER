const state = { catalog: [], leagues: [], models: {}, specs: [], jobs: new Map() };
const titles = {
  dashboard: "Inicio",
  leagues: "Ligas",
  data: "Datos",
  models: "Modelos",
  evaluate: "Evaluar",
  predict: "Predecir",
  analysis: "Analisis",
  config: "Configuracion",
};
const jobLabels = {
  queued: "En cola",
  running: "En ejecucion",
  succeeded: "Completado",
  failed: "Fallido",
};

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("view-meta").textContent = window.location.origin;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.getElementById("refresh-btn").addEventListener("click", refreshAll);
  bindForms();
  setDefaultDates();
  refreshAll();
  setInterval(pollJobs, 1800);
});

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!payload.ok) throw new Error(cleanMessage(payload.error || "Solicitud fallida"));
  return payload.data;
}

function switchView(view) {
  document.querySelectorAll(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  document.querySelectorAll(".view").forEach((panel) => panel.classList.toggle("active", panel.id === view));
  document.getElementById("view-title").textContent = titles[view] || view;
}

async function refreshAll() {
  clearAlert();
  try {
    const [dashboard, catalog, leagues, specs, config] = await Promise.all([
      api("/api/dashboard"),
      api("/api/leagues/catalog"),
      api("/api/leagues"),
      api("/api/model-specs"),
      api("/api/config/browser"),
    ]);
    state.leagues = leagues;
    document.getElementById("metric-leagues").textContent = dashboard.leagues;
    document.getElementById("metric-models").textContent = dashboard.models;
    fillCatalog(catalog);
    fillLeagueSelects(leagues);
    fillModelSpecs(specs);
    renderLeagues(leagues);
    fillConfig(config);
    if (leagues.length) {
      await refreshModelSelects();
      await loadData();
    }
  } catch (error) {
    showError(error.message);
  }
}

function bindForms() {
  document.getElementById("catalog-select").addEventListener("change", updateCatalogDefaults);
  const leagueIdInput = document.querySelector("#league-create-form input[name=league_id]");
  leagueIdInput.addEventListener("input", () => { leagueIdInput.dataset.autofilled = "false"; });
  document.getElementById("league-create-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitJson("/api/leagues", formJson(event.target), true);
  });
  document.getElementById("data-load").addEventListener("click", loadData);
  document.getElementById("data-league").addEventListener("change", loadData);
  document.getElementById("data-export").addEventListener("click", exportData);
  document.getElementById("models-load").addEventListener("click", loadModelsList);
  ["train-league", "model-type"].forEach((id) => {
    document.getElementById(id).addEventListener("change", updateTrainingDefaults);
  });
  const modelIdInput = document.querySelector("#train-form input[name=model_id]");
  modelIdInput.addEventListener("input", () => { modelIdInput.dataset.autofilled = "false"; });
  document.getElementById("tuning-enabled").addEventListener("change", toggleTuningControls);
  document.getElementById("train-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = formJson(event.target);
    const leagueId = payload.league_id;
    delete payload.league_id;
    await submitJson(`/api/leagues/${leagueId}/models/train`, payload, true);
  });
  ["eval-league", "manual-league", "fixtures-league"].forEach((id) => {
    document.getElementById(id).addEventListener("change", refreshModelSelects);
  });
  document.getElementById("evaluate-form").addEventListener("submit", evaluateModel);
  document.getElementById("manual-form").addEventListener("submit", manualPredict);
  document.getElementById("fixtures-form").addEventListener("submit", fixturesPredict);
  document.getElementById("analysis-form").addEventListener("submit", analysisPlot);
  document.getElementById("config-form").addEventListener("submit", saveConfig);
  toggleTuningControls();
}

async function submitJson(path, payload, jobExpected = false) {
  clearAlert();
  try {
    const result = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (jobExpected) trackJob(result);
    else showInfo("Listo");
    await refreshAll();
  } catch (error) {
    showError(error.message);
  }
}

function formJson(form) {
  const data = {};
  new FormData(form).forEach((value, key) => {
    if (value === "" && key !== "brave_binary") return;
    const field = form.elements[key];
    if (field && field.type === "checkbox") data[key] = field.checked;
    else if (field && field.type === "number") data[key] = Number(value);
    else data[key] = value;
  });
  form.querySelectorAll("input[type=checkbox]").forEach((input) => { data[input.name] = input.checked; });
  return data;
}

function fillCatalog(catalog) {
  state.catalog = catalog;
  document.getElementById("catalog-select").innerHTML = catalog.map((league) => `<option value="${league.index}">${escapeHtml(league.display_name)}</option>`).join("");
  updateCatalogDefaults();
}

function selectedCatalogLeague() {
  const selected = Number(document.getElementById("catalog-select").value);
  return state.catalog.find((league) => Number(league.index) === selected);
}

function updateCatalogDefaults() {
  const league = selectedCatalogLeague();
  const preview = document.getElementById("catalog-preview");
  const catalogFlag = document.getElementById("catalog-flag");
  if (!league) {
    preview.innerHTML = "";
    if (catalogFlag) catalogFlag.innerHTML = "";
    return;
  }

  if (catalogFlag) {
    catalogFlag.innerHTML = `<img class="flag" src="${escapeAttr(league.flag_url)}" alt="Bandera ${escapeAttr(league.country)}">`;
  }
  preview.innerHTML = `
    <img class="flag" src="${escapeAttr(league.flag_url)}" alt="Bandera ${escapeAttr(league.country)}">
    <div class="league-meta">
      <strong>${escapeHtml(league.display_name)}</strong>
      <small>${escapeHtml(league.category)} - desde ${escapeHtml(league.start_year)} - historial ${escapeHtml(league.history_window)} - margen ${escapeHtml(league.goal_margin)}</small>
    </div>`;

  const form = document.getElementById("league-create-form");
  const leagueId = form.elements.league_id;
  if (!leagueId.value || leagueId.dataset.autofilled !== "false") {
    leagueId.value = league.default_league_id;
    leagueId.dataset.autofilled = "true";
  }
  form.elements.start_year.value = league.start_year;
  form.elements.history_window.value = league.history_window;
  form.elements.goal_margin.value = league.goal_margin;
}

function fillLeagueSelects(leagues) {
  const html = leagues.map((league) => `<option value="${escapeAttr(league.league_id)}">${escapeHtml(league.league_id)} - ${escapeHtml(league.display_name)}</option>`).join("");
  ["data-league", "train-league", "models-league", "eval-league", "manual-league", "fixtures-league", "analysis-league"].forEach((id) => {
    document.getElementById(id).innerHTML = html;
  });
  updateTrainingDefaults();
}

function fillModelSpecs(specs) {
  state.specs = specs;
  document.getElementById("model-type").innerHTML = specs.map((spec) => `<option value="${escapeAttr(spec.key)}">${escapeHtml(spec.label)}</option>`).join("");
  updateTrainingDefaults();
}

function updateTrainingDefaults() {
  const form = document.getElementById("train-form");
  const leagueId = form.elements.league_id.value;
  const modelType = form.elements.model_type.value || "xgboost";
  const modelId = form.elements.model_id;
  const shortModel = { ngboost: "ngb", catboost: "cat", lightgbm: "lgbm", xgboost: "xgb" }[modelType] || modelType;
  if (!leagueId) return;
  if (!modelId.value || modelId.dataset.autofilled !== "false") {
    modelId.value = `${leagueId}-${shortModel}-result`;
    modelId.dataset.autofilled = "true";
  }
}

function toggleTuningControls() {
  const enabled = document.getElementById("tuning-enabled").checked;
  document.querySelectorAll("[data-tuning-control]").forEach((input) => {
    input.disabled = !enabled;
  });
}

function renderLeagues(leagues) {
  const target = document.getElementById("saved-leagues");
  if (!leagues.length) {
    target.innerHTML = `<div class="item"><div>No hay ligas guardadas</div></div>`;
    return;
  }
  target.innerHTML = `<div class="list">${leagues.map((league) => `
    <div class="item">
      <div class="league-main">
        <img class="flag" src="${escapeAttr(league.flag_url)}" alt="Bandera ${escapeAttr(league.country)}">
        <div class="league-meta">
          <strong>${escapeHtml(league.league_id)}</strong>
          <small>${escapeHtml(league.display_name)} - ${escapeHtml(league.rows)} filas - ${escapeHtml(league.models)} modelos - ${escapeHtml(league.stats)}</small>
        </div>
      </div>
      <div class="item-actions">
        <button onclick="updateLeague('${escapeAttr(league.league_id)}')">Actualizar</button>
        <button class="danger" onclick="deleteLeague('${escapeAttr(league.league_id)}')">Eliminar</button>
      </div>
    </div>`).join("")}</div>`;
}

async function updateLeague(leagueId) {
  await submitJson(`/api/leagues/${leagueId}/update`, {}, true);
}

async function deleteLeague(leagueId) {
  if (!confirm(`Eliminar ${leagueId}?`)) return;
  try {
    await api(`/api/leagues/${leagueId}`, { method: "DELETE" });
    await refreshAll();
  } catch (error) {
    showError(error.message);
  }
}

async function loadData() {
  const league = document.getElementById("data-league").value;
  if (!league) return;
  const params = new URLSearchParams({
    query: document.getElementById("data-query").value,
    column: document.getElementById("data-column").value,
    hide_missing: document.getElementById("data-missing").checked,
    page_size: 100,
  });
  try {
    const data = await api(`/api/leagues/${league}/data?${params}`);
    renderTable("data-table", data);
  } catch (error) {
    showError(error.message);
  }
}

async function exportData() {
  const league = document.getElementById("data-league").value;
  if (!league) return;
  try {
    const data = await api(`/api/leagues/${league}/data/export?fmt=csv`);
    window.open(data.url, "_blank");
  } catch (error) {
    showError(error.message);
  }
}

async function refreshModelSelects() {
  const leagueIds = [...new Set([
    document.getElementById("eval-league").value,
    document.getElementById("manual-league").value,
    document.getElementById("fixtures-league").value,
    document.getElementById("models-league").value,
  ].filter(Boolean))];
  await Promise.all(leagueIds.map(loadModelsForLeague));
  fillModelSelect("eval-model", document.getElementById("eval-league").value);
  fillModelSelect("manual-model", document.getElementById("manual-league").value);
  fillModelSelect("fixtures-model", document.getElementById("fixtures-league").value);
  await loadModelsList();
}

async function loadModelsForLeague(leagueId) {
  if (!leagueId) return;
  state.models[leagueId] = await api(`/api/leagues/${leagueId}/models`);
}

function fillModelSelect(id, leagueId) {
  const models = state.models[leagueId] || [];
  document.getElementById(id).innerHTML = models.map((model) => `<option value="${escapeAttr(model.model_id)}">${escapeHtml(model.model_id)}</option>`).join("");
}

async function loadModelsList() {
  const league = document.getElementById("models-league").value;
  if (!league) return;
  try {
    await loadModelsForLeague(league);
    const models = state.models[league] || [];
    document.getElementById("models-list").innerHTML = `<div class="list">${models.map((model) => `
      <div class="item">
        <div><strong>${escapeHtml(model.model_id)}</strong><br><small>${escapeHtml(model.class || "Modelo")} - ${escapeHtml(model.target)}</small></div>
        <button class="danger" onclick="deleteModel('${escapeAttr(league)}', '${escapeAttr(model.model_id)}')">Eliminar</button>
      </div>`).join("") || `<div class="item"><div>Sin modelos</div></div>`}</div>`;
  } catch (error) {
    showError(error.message);
  }
}

async function deleteModel(league, modelId) {
  if (!confirm(`Eliminar ${modelId}?`)) return;
  try {
    await api(`/api/leagues/${league}/models/${modelId}`, { method: "DELETE" });
    await refreshAll();
  } catch (error) {
    showError(error.message);
  }
}

async function evaluateModel(event) {
  event.preventDefault();
  const payload = formJson(event.target);
  const league = payload.league_id;
  const model = payload.model_id;
  delete payload.league_id;
  delete payload.model_id;
  try {
    const result = await api(`/api/leagues/${league}/models/${model}/evaluate`, jsonOptions(payload));
    document.getElementById("evaluate-output").innerHTML = tableHtml(result.metrics) + tableHtml(result.rows);
  } catch (error) {
    showError(error.message);
  }
}

async function manualPredict(event) {
  event.preventDefault();
  const payload = formJson(event.target);
  const league = payload.league_id;
  delete payload.league_id;
  try {
    const result = await api(`/api/leagues/${league}/predict/manual`, jsonOptions(payload));
    renderTable("predict-output", result.prediction);
  } catch (error) {
    showError(error.message);
  }
}

async function fixturesPredict(event) {
  event.preventDefault();
  const form = event.target;
  const data = new FormData(form);
  const league = data.get("league_id");
  data.delete("league_id");
  try {
    const result = await api(`/api/leagues/${league}/predict/fixtures`, { method: "POST", body: data });
    trackJob(result);
  } catch (error) {
    showError(error.message);
  }
}

async function analysisPlot(event) {
  event.preventDefault();
  const payload = formJson(event.target);
  const league = payload.league_id;
  const type = payload.analysis_type;
  delete payload.league_id;
  delete payload.analysis_type;
  await submitJson(`/api/leagues/${league}/analysis/${type}`, payload, true);
}

async function saveConfig(event) {
  event.preventDefault();
  try {
    await api("/api/config/browser", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(formJson(event.target)) });
    showInfo("Config guardada");
  } catch (error) {
    showError(error.message);
  }
}

function fillConfig(config) {
  const form = document.getElementById("config-form");
  form.elements.application.value = config.application;
  form.elements.brave_binary.value = config.brave_binary || "";
  form.elements.headless.checked = Boolean(config.headless);
}

async function pollJobs() {
  if (!state.jobs.size) return;
  for (const jobId of [...state.jobs.keys()]) {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      state.jobs.set(jobId, job);
      if (job.status === "succeeded" || job.status === "failed") {
        if (job.result && job.result.image) renderImage("analysis-output", job.result.image.url);
        if (job.result && job.result.predictions) renderTable("predict-output", job.result.predictions);
        if (job.result && job.result.results) renderTrainingResult(job.result);
        if (job.status === "succeeded") await refreshAll();
      }
    } catch (error) {
      state.jobs.delete(jobId);
    }
  }
  renderJobs();
}

function trackJob(job) {
  state.jobs.set(job.job_id, job);
  renderJobs();
  showInfo(`Proceso ${job.job_id} en cola`);
}

function renderJobs() {
  const target = document.getElementById("jobs-list");
  const jobs = [...state.jobs.values()].slice(-8).reverse();
  target.innerHTML = jobs.map((job) => `<div class="job ${escapeAttr(job.status)}"><strong>${escapeHtml(jobLabels[job.status] || job.status)}</strong> - ${escapeHtml(job.message)}${jobProgressHtml(job)}<br><small>${escapeHtml(job.job_id)}${job.error ? " - " + escapeHtml(cleanMessage(job.error)) : ""}</small></div>`).join("") || "<div class='item'><div>Sin procesos activos</div></div>";
}

function jobProgressHtml(job) {
  const progress = job.progress || {};
  if (progress.stage !== "tuning") return "";
  const current = progress.current_trial || 0;
  const total = progress.total_trials || 0;
  const best = progress.best_value === null || progress.best_value === undefined ? "" : ` - mejor ${Number(progress.best_value).toFixed(3)}`;
  return ` - Optuna ${escapeHtml(current)}/${escapeHtml(total)}${escapeHtml(best)}`;
}

function renderTrainingResult(result) {
  const target = document.getElementById("training-output");
  const model = result.model || {};
  const optuna = result.optuna || {};
  const results = result.results || {};
  const blocks = [
    `<div class="output-block">
      <h2>${escapeHtml(model.model_id || "Modelo")}</h2>
      <div class="param-grid">
        <div class="param"><span>Tipo</span>${escapeHtml(model.class || "")}</div>
        <div class="param"><span>Objetivo</span>${escapeHtml(model.target || "")}</div>
        <div class="param"><span>Eval %</span>${escapeHtml(model.eval_size || "")}</div>
      </div>
    </div>`,
  ];
  if (optuna.enabled) {
    blocks.push(`<div class="output-block">
      <h2>Optuna</h2>
      <div class="param-grid">
        <div class="param"><span>Sampler</span>${escapeHtml(optuna.sampler)}</div>
        <div class="param"><span>Pruner</span>${escapeHtml(optuna.pruner)}</div>
        <div class="param"><span>Trials</span>${escapeHtml(optuna.n_trials)}</div>
        <div class="param"><span>Mejor score</span>${escapeHtml(formatNumber(optuna.best_score))}</div>
      </div>
      ${renderParamGrid(optuna.best_params || {})}
      ${renderOptunaChart(results.tune, optuna.objective)}
      ${tableHtml(results.tune)}
    </div>`);
  }
  if (results.fit) blocks.push(`<div class="output-block"><h2>Entrenamiento</h2>${tableHtml(results.fit)}</div>`);
  if (results.cv) blocks.push(`<div class="output-block"><h2>Validacion cruzada</h2>${tableHtml(results.cv)}</div>`);
  if (results["sliding-cv"]) blocks.push(`<div class="output-block"><h2>CV deslizante</h2>${tableHtml(results["sliding-cv"])}</div>`);
  target.innerHTML = blocks.join("");
}

function renderParamGrid(params) {
  const entries = Object.entries(params);
  if (!entries.length) return "";
  return `<div class="param-grid">${entries.map(([key, value]) => `<div class="param"><span>${escapeHtml(key)}</span>${escapeHtml(formatNumber(value))}</div>`).join("")}</div>`;
}

function renderOptunaChart(table, metric) {
  if (!table || !table.rows || !metric) return "";
  const points = table.rows
    .map((row, index) => ({ trial: Number(row.Trial ?? index), value: Number(row[metric]) }))
    .filter((point) => Number.isFinite(point.trial) && Number.isFinite(point.value))
    .sort((a, b) => a.trial - b.trial);
  if (points.length < 2) return "";
  const width = 720;
  const height = 180;
  const pad = 24;
  const minX = Math.min(...points.map((point) => point.trial));
  const maxX = Math.max(...points.map((point) => point.trial));
  const minY = Math.min(...points.map((point) => point.value));
  const maxY = Math.max(...points.map((point) => point.value));
  const scaleX = (value) => pad + ((value - minX) / Math.max(maxX - minX, 1)) * (width - pad * 2);
  const scaleY = (value) => height - pad - ((value - minY) / Math.max(maxY - minY, 0.001)) * (height - pad * 2);
  const polyline = points.map((point) => `${scaleX(point.trial).toFixed(1)},${scaleY(point.value).toFixed(1)}`).join(" ");
  const circles = points.map((point) => `<circle cx="${scaleX(point.trial).toFixed(1)}" cy="${scaleY(point.value).toFixed(1)}" r="3"></circle>`).join("");
  return `<svg class="optuna-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Optuna">
    <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d9e1ea"></line>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d9e1ea"></line>
    <polyline points="${polyline}" fill="none" stroke="#2563eb" stroke-width="2.5"></polyline>
    <g fill="#2563eb">${circles}</g>
  </svg>`;
}

function renderTable(id, table) {
  document.getElementById(id).innerHTML = tableHtml(table);
}

function tableHtml(table) {
  if (!table || !table.columns) return "<div></div>";
  const head = table.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const rows = table.rows.map((row) => `<tr>${table.columns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div><small>${table.total} filas</small>`;
}

function renderImage(id, url) {
  document.getElementById(id).innerHTML = `<img class="output-image" src="${escapeAttr(url)}" alt="resultado">`;
}

function jsonOptions(payload) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
}

function showInfo(message) {
  const alert = document.getElementById("alert");
  alert.textContent = message;
  alert.className = "alert";
}

function showError(message) {
  const alert = document.getElementById("alert");
  alert.textContent = cleanMessage(message);
  alert.className = "alert error";
}

function clearAlert() {
  const alert = document.getElementById("alert");
  alert.textContent = "";
  alert.className = "alert hidden";
}

function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char]));
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function cleanMessage(message) {
  return String(message || "")
    .replace(/^(CLIError|ValueError|RuntimeError|NotImplementedError):\s*/, "")
    .replace(/\bNone\b/g, "Sin valor");
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : value;
}

function setDefaultDates() {
  const dateInput = document.querySelector("#fixtures-form input[name=date]");
  if (dateInput && !dateInput.value) dateInput.value = new Date().toISOString().slice(0, 10);
}
