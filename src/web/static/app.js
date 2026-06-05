const state = { leagues: [], models: {}, jobs: new Map() };
const titles = {
  dashboard: "Inicio",
  leagues: "Ligas",
  data: "Datos",
  models: "Modelos",
  evaluate: "Evaluar",
  predict: "Predecir",
  analysis: "Analisis",
  config: "Config",
};

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("view-meta").textContent = window.location.origin;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.getElementById("refresh-btn").addEventListener("click", refreshAll);
  bindForms();
  refreshAll();
  setInterval(pollJobs, 1800);
});

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!payload.ok) throw new Error(payload.error || "Request failed");
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
    if (leagues.length) await refreshModelSelects();
  } catch (error) {
    showError(error.message);
  }
}

function bindForms() {
  document.getElementById("league-create-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitJson("/api/leagues", formJson(event.target), true);
  });
  document.getElementById("data-load").addEventListener("click", loadData);
  document.getElementById("data-export").addEventListener("click", exportData);
  document.getElementById("models-load").addEventListener("click", loadModelsList);
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
    if (value === "") return;
    const field = form.elements[key];
    if (field && field.type === "checkbox") data[key] = field.checked;
    else if (field && field.type === "number") data[key] = Number(value);
    else data[key] = value;
  });
  form.querySelectorAll("input[type=checkbox]").forEach((input) => { data[input.name] = input.checked; });
  return data;
}

function fillCatalog(catalog) {
  document.getElementById("catalog-select").innerHTML = catalog.map((league) => `<option value="${league.index}">${league.country} / ${league.name}</option>`).join("");
}

function fillLeagueSelects(leagues) {
  const html = leagues.map((league) => `<option value="${league.league_id}">${league.league_id}</option>`).join("");
  ["data-league", "train-league", "models-league", "eval-league", "manual-league", "fixtures-league", "analysis-league"].forEach((id) => {
    document.getElementById(id).innerHTML = html;
  });
}

function fillModelSpecs(specs) {
  document.getElementById("model-type").innerHTML = specs.map((spec) => `<option value="${spec.key}">${spec.label}</option>`).join("");
}

function renderLeagues(leagues) {
  const target = document.getElementById("saved-leagues");
  if (!leagues.length) {
    target.innerHTML = `<div class="item"><div>No hay ligas guardadas</div></div>`;
    return;
  }
  target.innerHTML = `<div class="list">${leagues.map((league) => `
    <div class="item">
      <div><strong>${league.league_id}</strong><br><small>${league.country} / ${league.name} - ${league.rows} filas - ${league.models} modelos</small></div>
      <div>
        <button onclick="updateLeague('${league.league_id}')">R</button>
        <button class="danger" onclick="deleteLeague('${league.league_id}')">X</button>
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
  document.getElementById(id).innerHTML = models.map((model) => `<option value="${model.model_id}">${model.model_id}</option>`).join("");
}

async function loadModelsList() {
  const league = document.getElementById("models-league").value;
  if (!league) return;
  try {
    await loadModelsForLeague(league);
    const models = state.models[league] || [];
    document.getElementById("models-list").innerHTML = `<div class="list">${models.map((model) => `
      <div class="item">
        <div><strong>${model.model_id}</strong><br><small>${model.class || "Model"} - ${model.target}</small></div>
        <button class="danger" onclick="deleteModel('${league}', '${model.model_id}')">X</button>
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
  showInfo(`Job ${job.job_id} en cola`);
}

function renderJobs() {
  const target = document.getElementById("jobs-list");
  const jobs = [...state.jobs.values()].slice(-8).reverse();
  target.innerHTML = jobs.map((job) => `<div class="job ${job.status}"><strong>${job.status}</strong> - ${job.message}<br><small>${job.job_id}${job.error ? " - " + job.error : ""}</small></div>`).join("") || "<div class='item'><div>Sin jobs activos</div></div>";
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
  document.getElementById(id).innerHTML = `<img class="output-image" src="${url}" alt="output">`;
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
  alert.textContent = message;
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
