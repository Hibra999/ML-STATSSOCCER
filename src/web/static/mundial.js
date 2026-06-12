const state = {
  overview: null,
  groups: [],
  fixtures: [],
  teams: [],
  players: [],
  models: [],
  activeModelId: "",
  training: null,
  trainingOptions: null,
  teamAssets: new Map(),
  defaultsApplied: false,
  trainingControlsApplied: false,
  countdownTimer: null,
  jobs: new Map(),
  jobTimer: null,
  jobPollingInFlight: false,
  newModelMode: false,
  lastSimulation: null,
  lastUpcomingReport: null,
};

const jobLabels = {
  queued: "En cola",
  running: "En proceso",
  succeeded: "Completado",
  failed: "Error",
};

const goalMarketLines = [
  { key: "over_under_05", label: "U/O 0.5", over: "over05", under: "under05" },
  { key: "over_under_15", label: "U/O 1.5", over: "over15", under: "under15" },
  { key: "over_under_25", label: "U/O 2.5", over: "over25", under: "under25" },
  { key: "over_under_35", label: "U/O 3.5", over: "over35", under: "under35" },
];

const trainingMarketOrder = ["result", "over_under_05", "over_under_15", "over_under_25", "over_under_35", "goals_distribution"];
const poissonRecentInputIds = ["sim-poisson-recent-matches", "upcoming-poisson-recent-matches"];

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadAll(false);
});

function bindEvents() {
  document.querySelectorAll("[data-section]").forEach((button) => {
    button.addEventListener("click", () => switchWorldcupView(button.dataset.section));
  });
  document.getElementById("refresh-btn").addEventListener("click", () => loadAll(true));
  document.getElementById("simulate-poisson-btn").addEventListener("click", runMatchMonteCarlo);
  document.getElementById("worldcup-new-model").addEventListener("click", startNewWorldcupModel);
  document.getElementById("model-load").addEventListener("click", loadSelectedModel);
  document.getElementById("model-delete").addEventListener("click", deleteSelectedModel);
  document.getElementById("worldcup-clear-cache").addEventListener("click", clearWorldcupMaintenance);
  document.getElementById("model-active-select").addEventListener("change", syncModelSelects);
  document.getElementById("upcoming-model-select").addEventListener("change", syncModelSelects);
  document.getElementById("fixture-group-filter").addEventListener("change", renderFixtures);
  document.getElementById("fixture-search").addEventListener("input", renderFixtures);
  document.getElementById("players-refresh").addEventListener("click", () => loadPlayers(true));
  document.getElementById("training-refresh").addEventListener("click", loadTrainingStatus);
  document.getElementById("training-download").addEventListener("click", downloadTrainingDataset);
  document.getElementById("training-prepare-etl").addEventListener("click", prepareTrainingEtl);
  document.getElementById("training-refresh-snapshots").addEventListener("click", refreshPlayerSnapshots);
  document.getElementById("training-train").addEventListener("click", trainWorldCupModel);
  document.getElementById("training-retrain-base").addEventListener("click", () => trainWorldCupModel("result_only"));
  document.getElementById("upcoming-predict-btn").addEventListener("click", runUpcomingPredictions);
  document.getElementById("upcoming-pipeline-mode").addEventListener("change", syncUpcomingPipelineControls);
  document.getElementById("worldcup-model-type").addEventListener("change", () => applyModelDefaults(document.getElementById("worldcup-model-type").value, true));
  document.getElementById("worldcup-model-id").addEventListener("input", (event) => { event.target.dataset.autofilled = "false"; });
  document.getElementById("worldcup-tuning-enabled").addEventListener("change", applyTuningLocks);
  document.getElementById("worldcup-tune-params").addEventListener("input", applyTuningLocks);
  poissonRecentInputIds.forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.addEventListener("change", () => syncPoissonRecentInputs(input));
  });
  syncUpcomingPipelineControls();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!payload.ok) throw new Error(cleanMessage(payload.error || "Solicitud fallida"));
  return payload.data;
}

async function loadAll(refresh) {
  clearAlert();
  setLoading();
  try {
    const [overview, groups, fixtures, teams, players, training, models, procedure] = await Promise.all([
      api(`/api/mundial/overview?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/groups?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/fixtures?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/teams?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/players?refresh=${refresh ? "true" : "false"}`),
      api("/api/mundial/training/status"),
      api("/api/mundial/models"),
      api("/api/mundial/procedure"),
    ]);
    state.overview = overview;
    state.groups = groups.groups || [];
    state.fixtures = fixtures.fixtures || [];
    state.teams = teams.teams || [];
    state.players = players.players || [];
    state.training = training;
    state.models = models.models || [];
    state.activeModelId = models.active_model_id || "";
    state.trainingOptions = training.options || null;
    state.lastSimulation = overview.last_simulation || state.lastSimulation;
    rebuildTeamAssets();
    renderScoreModelOptions(overview.score_models || []);
    applyDefaultConfig(overview.default_config || {});
    renderOverview(overview);
    renderGroups(groups);
    renderTeams(teams);
    renderFixtureFilters();
    renderFixtures();
    fillUpcomingGroupFilter();
    fillSimulationGroupFilter();
    renderPlayers(players);
    renderTrainingStatus(training);
    renderModelsCatalog(models);
    renderProcedure(procedure);
  } catch (error) {
    showError(error.message);
  }
}

function setLoading() {
  document.getElementById("groups-grid").innerHTML = loadingHtml("Cargando grupos");
  document.getElementById("teams-grid").innerHTML = loadingHtml("Cargando equipos");
  document.getElementById("fixtures-list").innerHTML = loadingHtml("Cargando fixtures");
  document.getElementById("players-list").innerHTML = loadingHtml("Cargando jugadores");
  document.getElementById("training-summary").innerHTML = loadingHtml("Dataset pendiente");
  document.getElementById("training-model-state").innerHTML = loadingHtml("Modelo pendiente");
  document.getElementById("training-etl-flow").innerHTML = loadingHtml("ETL pendiente");
  document.getElementById("training-metric-cards").innerHTML = loadingHtml("Metricas pendientes");
  document.getElementById("training-confusion-matrix").innerHTML = loadingHtml("Matriz pendiente");
  document.getElementById("training-tuning-flow").innerHTML = loadingHtml("Tuning pendiente");
  document.getElementById("training-features").innerHTML = loadingHtml("Features pendientes");
  document.getElementById("training-model-params").innerHTML = loadingHtml("Parametros pendientes");
  renderWorldcupJobProgress("training");
  document.getElementById("upcoming-predictions").innerHTML = loadingHtml("Predicciones pendientes");
  document.getElementById("upcoming-report").innerHTML = loadingHtml("Reporte pendiente");
  renderWorldcupJobProgress("upcoming-report");
  document.getElementById("match-simulation-grid").innerHTML = loadingHtml("Monte Carlo pendiente");
  document.getElementById("match-simulation-table").innerHTML = "";
  document.getElementById("active-model-state").innerHTML = loadingHtml("Modelo pendiente");
  document.getElementById("models-list").innerHTML = loadingHtml("Modelos pendientes");
  document.getElementById("simulation-summary").innerHTML = "";
  renderWorldcupJobProgress("simulation");
}

function applyDefaultConfig(config) {
  if (state.defaultsApplied) return;
  const pairs = {
    "sim-iterations": config.iterations,
    "sim-seed": config.seed,
    "sim-history-weight": config.history_weight,
    "sim-recency-weight": config.recency_weight,
    "sim-host-advantage": config.host_advantage,
    "sim-max-goals": config.max_goals,
    "sim-ml-weight": config.ml_weight,
    "sim-poisson-recent-matches": config.poisson_recent_matches,
    "upcoming-poisson-recent-matches": config.poisson_recent_matches,
    "sim-score-model": config.score_model,
  };
  Object.entries(pairs).forEach(([id, value]) => {
    const input = document.getElementById(id);
    if (input && value !== undefined) input.value = value;
  });
  const simMlToggle = document.getElementById("sim-use-ml-model");
  if (simMlToggle) simMlToggle.checked = Boolean(config.use_ml_model);
  state.defaultsApplied = true;
}

function renderScoreModelOptions(options) {
  const select = document.getElementById("sim-score-model");
  if (!select) return;
  const rows = options.length ? options : [{ key: "independent_poisson", label: "Poisson independiente" }];
  select.innerHTML = rows.map((option) => (
    `<option value="${escapeAttr(option.key || "")}">${escapeHtml(option.label || option.key || "")}</option>`
  )).join("");
}

function renderOverview(overview) {
  document.getElementById("metric-teams").textContent = overview.teams || 0;
  document.getElementById("metric-groups").textContent = overview.groups || 0;
  document.getElementById("metric-fixtures").textContent = overview.fixtures || 0;
  document.getElementById("metric-results").textContent = overview.confirmed_results || 0;
  const resultSource = overview.result_source ? ` · resultados: ${overview.result_source}` : "";
  document.getElementById("model-source").textContent = `${overview.model || "Modelo"} - ${overview.fixture_source || ""}${resultSource}`;
  const featured = overview.featured_matches || [];
  const highlight = overview.highlight || featured[0] || {};
  document.getElementById("hero-meta").textContent = featured.length
    ? `${featured.length} partido${featured.length === 1 ? "" : "s"} en el próximo kickoff`
    : "Sin próximos partidos futuros";
  document.getElementById("hero-match").innerHTML = featured.length
    ? featured.map((fixture, index) => heroFeaturedMatchHtml(fixture, index === 0)).join("")
    : `<article class="hero-featured-card empty"><strong>Sin próximos partidos</strong><small>Todos los partidos cargados ya iniciaron o finalizaron.</small></article>`;
  renderHeroCountdown(overview.countdown_target, overview.countdown_state, highlight);
  renderHeroHardware((state.trainingOptions || {}).hardware || {});
  document.getElementById("overview-next-source").textContent = overview.fixture_source || "";
  document.getElementById("overview-standings-source").textContent = overview.result_source || "fixture-cache";
  document.getElementById("hero-next-grid").innerHTML = (overview.next_matches || []).map((fixture) => heroNextCardHtml(fixture)).join("")
    || `<article class="hero-next-card empty"><strong>Sin más partidos cargados</strong><small>El calendario adicional aparecerá aquí.</small></article>`;
  renderOverviewStandings(overview.group_standings || [], overview);
  renderQuickSimulationPanel(state.lastSimulation);
}

function matchTeamHtml(asset, side) {
  return `<div class="match-team ${escapeAttr(side)}">
    ${side === "away" ? `<strong>${escapeHtml(asset.name || "")}</strong>` : ""}
    ${flagHtml(asset, "large")}
    ${side !== "away" ? `<strong>${escapeHtml(asset.name || "")}</strong>` : ""}
  </div>`;
}

function heroFeaturedMatchHtml(fixture, withCountdown) {
  const kickoffLabel = `${fixture.date || ""} ${fixture.time || ""}`.trim();
  return `<article class="hero-featured-card ${withCountdown ? "featured" : ""}">
    ${matchTeamHtml(fixture.home || {}, "home")}
    <div class="hero-vs-block">
      <span class="versus">VS</span>
      <div class="hero-kickoff">
        <strong>${escapeHtml(kickoffLabel || "Horario pendiente")}</strong>
        <small>${escapeHtml(fixture.venue || "Sede por confirmar")}</small>
      </div>
      <small>${escapeHtml(fixture.group || fixture.round || "")}</small>
    </div>
    ${matchTeamHtml(fixture.away || {}, "away")}
  </article>`;
}

function renderOverviewStandings(groups, overview = {}) {
  const source = overview.result_source || "fixture-cache";
  const updated = overview.results_updated_at ? `Actualizado ${overview.results_updated_at}` : "Sin hora de actualizacion";
  const highlightGroup = (overview.highlight || {}).group || "";
  const candidates = groups || [];
  const selected = candidates.find((group) => Number(group.played_matches || 0) > 0)
    || candidates.find((group) => group.name === highlightGroup)
    || candidates.find((group) => group.letter === "A")
    || candidates[0];
  if (!selected) {
    document.getElementById("overview-standings").innerHTML = loadingHtml("Grupo pendiente");
    return;
  }
  document.getElementById("overview-standings").innerHTML = `
    <article class="overview-standing-card">
      <header><strong>${escapeHtml(selected.name || selected.letter || "")}</strong><span>PJ</span><span>Pts</span><span>DG</span></header>
      ${(selected.rows || []).map((row) => `
        <div>
          <span>${flagHtml(row)}${escapeHtml(row.team || row.name || "")}</span>
          <small>${escapeHtml(row.PJ ?? 0)}</small>
          <b>${escapeHtml(row.Pts ?? 0)}</b>
          <small>${escapeHtml(row.DG ?? 0)}</small>
        </div>`).join("")}
      <footer>${escapeHtml(source)} · ${escapeHtml(updated)}</footer>
    </article>`;
}

function renderGroups(payload) {
  const sourceParts = [
    payload.source || "",
    payload.result_source ? `resultados: ${payload.result_source}` : "",
    payload.results_updated_at ? `actualizado ${payload.results_updated_at}` : "",
  ].filter(Boolean);
  document.getElementById("groups-source").textContent = sourceParts.join(" · ");
  document.getElementById("groups-grid").innerHTML = state.groups.map((group) => `
    <article class="group-card ${Number(group.played_matches || 0) > 0 ? "has-results" : ""}">
      <header><h3>${escapeHtml(group.name)}</h3><strong>${escapeHtml(group.letter)}</strong></header>
      <table class="standings-table">
        <thead><tr><th>Equipo</th><th>PJ</th><th>GF</th><th>GC</th><th>DG</th><th>Pts</th></tr></thead>
        <tbody>${(group.standings || []).map((team) => `<tr>
          <td>${flagHtml(team)}<strong>${escapeHtml(team.team || team.name)}</strong></td>
          <td>${escapeHtml(team.PJ ?? 0)}</td>
          <td>${escapeHtml(team.GF ?? 0)}</td>
          <td>${escapeHtml(team.GC ?? 0)}</td>
          <td>${escapeHtml(team.DG ?? 0)}</td>
          <td><b>${escapeHtml(team.Pts ?? 0)}</b></td>
        </tr>`).join("")}</tbody>
      </table>
    </article>`).join("");
}

function renderTeams(payload) {
  const rows = payload.teams || state.teams;
  document.getElementById("teams-grid").innerHTML = rows.map((row) => {
    const asset = row.asset || assetFor(row.Equipo);
    return `<article class="team-card">
      <header>${flagHtml(asset, "large")}<div><strong>${escapeHtml(asset.name)}</strong><small>${escapeHtml(row.Grupo || "")}${row.is_host ? " - Sede" : ""}</small></div></header>
      <div class="team-stats">
        <div class="team-stat"><span>Rating</span><strong>${escapeHtml(row.Rating)}</strong></div>
        <div class="team-stat"><span>Ataque</span><strong>${escapeHtml(row.Ataque)}</strong></div>
        <div class="team-stat"><span>Defensa</span><strong>${escapeHtml(row["Defensa rival"])}</strong></div>
      </div>
    </article>`;
  }).join("");
}

function renderFixtureFilters() {
  const groups = [...new Set(state.fixtures.map((fixture) => fixture.group).filter(Boolean))];
  document.getElementById("fixture-group-filter").innerHTML = `<option value="">Todos los grupos</option>${groups.map((group) => `<option value="${escapeAttr(group)}">${escapeHtml(group)}</option>`).join("")}`;
}

function renderFixtures() {
  const group = document.getElementById("fixture-group-filter").value;
  const query = document.getElementById("fixture-search").value.trim().toLowerCase();
  const fixtures = state.fixtures.filter((fixture) => {
    if (group && fixture.group !== group) return false;
    if (!query) return true;
    return `${fixture.home.name} ${fixture.away.name}`.toLowerCase().includes(query);
  });
  document.getElementById("fixtures-list").innerHTML = fixtures.map((fixture) => `
    <article class="fixture-card">
      <div class="fixture-meta"><span>${escapeHtml(fixture.date)} ${escapeHtml(fixture.time || "")}</span><span>${escapeHtml(fixture.group || fixture.round)}</span></div>
      <div class="fixture-teams">
        <div class="fixture-team">${flagHtml(fixture.home)}<strong>${escapeHtml(fixture.home.name)}</strong></div>
        <span>${fixture.finished ? `${escapeHtml(fixture.score_home)}-${escapeHtml(fixture.score_away)}` : "vs"}</span>
        <div class="fixture-team">${flagHtml(fixture.away)}<strong>${escapeHtml(fixture.away.name)}</strong></div>
      </div>
      <small>${escapeHtml(fixture.venue || "Sede por confirmar")}</small>
    </article>`).join("") || loadingHtml("Sin fixtures para ese filtro");
}

function fillUpcomingGroupFilter() {
  const groups = [...new Set(state.fixtures.map((fixture) => fixture.group).filter(Boolean))];
  document.getElementById("upcoming-group-filter").innerHTML = `<option value="">Todos los grupos</option>${groups.map((group) => `<option value="${escapeAttr(group)}">${escapeHtml(group)}</option>`).join("")}`;
}

function fillSimulationGroupFilter() {
  const groups = [...new Set(state.fixtures.map((fixture) => fixture.group).filter(Boolean))];
  const select = document.getElementById("sim-group-filter");
  if (!select) return;
  select.innerHTML = `<option value="">Todos los grupos</option>${groups.map((group) => `<option value="${escapeAttr(group)}">${escapeHtml(group)}</option>`).join("")}`;
}

function renderHeroCountdown(targetIso, stateLabel, highlight) {
  const container = document.getElementById("hero-countdown");
  if (!container) return;
  if (state.countdownTimer) {
    window.clearInterval(state.countdownTimer);
    state.countdownTimer = null;
  }
  const matchLine = highlight && highlight.match ? highlight.match : "Horario por confirmar";
  const kickoffLine = [highlight && highlight.date, highlight && highlight.time, highlight && highlight.venue].filter(Boolean).join(" · ");
  if (!targetIso) {
    const label = stateLabel === "finished" ? "Finalizado" : "Sin horario";
    container.innerHTML = dashboardCountdownHtml(label, matchLine, kickoffLine, null);
    return;
  }
  const render = () => {
    const diff = Date.parse(targetIso) - Date.now();
    if (Number.isNaN(diff)) {
      container.innerHTML = dashboardCountdownHtml("Sin horario", matchLine, kickoffLine, null);
      return;
    }
    if (diff <= 0) {
      container.innerHTML = dashboardCountdownHtml("En curso", matchLine, kickoffLine, null);
      return;
    }
    const remaining = countdownParts(diff);
    container.innerHTML = dashboardCountdownHtml("Próximo", matchLine, kickoffLine, remaining);
  };
  render();
  state.countdownTimer = window.setInterval(render, 1000);
}

function dashboardCountdownHtml(label, matchLine, kickoffLine, remaining) {
  const cells = remaining
    ? [
      countdownChip("Días", remaining.days),
      countdownChip("Horas", remaining.hours),
      countdownChip("Min", remaining.minutes),
      countdownChip("Seg", remaining.seconds),
    ].join("")
    : `<div class="countdown-chip live"><span>Estado</span><strong>${escapeHtml(label)}</strong></div>`;
  return `
    <div class="countdown-head">
      <span>${escapeHtml(label)}</span>
      <strong id="hero-countdown-vs">${escapeHtml(matchLine || "Partido pendiente")}</strong>
      <small>${escapeHtml(kickoffLine || "Horario pendiente")}</small>
    </div>
    <div class="hero-countdown">${cells}</div>`;
}

function countdownParts(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return { days, hours: pad2(hours), minutes: pad2(minutes), seconds: pad2(seconds) };
}

function countdownChip(label, value) {
  return `<div class="countdown-chip"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function heroNextCardHtml(fixture) {
  return `<article class="hero-next-card">
    <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || fixture.round || "")}</strong></header>
    <div class="fixture-teams">
      <div class="fixture-team">${flagHtml(fixture.home)}<strong>${escapeHtml((fixture.home || {}).name || "")}</strong></div>
      <span>vs</span>
      <div class="fixture-team">${flagHtml(fixture.away)}<strong>${escapeHtml((fixture.away || {}).name || "")}</strong></div>
    </div>
    <small>${escapeHtml([fixture.time || "", fixture.venue || ""].filter(Boolean).join(" - ") || "Sede pendiente")}</small>
  </article>`;
}

function renderHeroHardware(hardware) {
  const container = document.getElementById("hero-hardware");
  if (!container) return;
  container.innerHTML = [
    hardwareChip("Device", hardware.actual_device || hardware.device_default || "cpu", "Motor"),
    hardwareChip("CUDA", hardware.cuda_available ? "Si" : "No", hardware.cuda_available ? "GPU disponible" : "CPU fallback", hardware.cuda_available ? "ok" : "warn"),
    hardwareChip("CPU", hardware.cpu_count || "-", "nucleos"),
    hardwareChip("Threads", hardware.effective_n_jobs || hardware.n_jobs || hardware.default_n_jobs || "-", "n_jobs"),
  ].join("");
}

function hardwareChip(label, value, detail, status) {
  return `<div class="hardware-chip ${status ? `hardware-${escapeAttr(status)}` : ""}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail || "")}</small></div>`;
}

async function loadModelsCatalog() {
  const result = await api("/api/mundial/models");
  state.models = result.models || [];
  state.activeModelId = result.active_model_id || "";
  renderModelsCatalog(result);
  return result;
}

function renderModelsCatalog(payload) {
  const models = (payload && payload.models) || state.models || [];
  const activeId = (payload && payload.active_model_id) || state.activeModelId || "";
  state.models = models;
  state.activeModelId = activeId;
  const options = models.map((model) => `<option value="${escapeAttr(model.model_id)}">${escapeHtml(model.model_name || model.model_id)}${model.bundle ? " - 1X2 + U/O ML" : ""}${model.active ? " - activo" : ""}</option>`).join("");
  const selectHtml = `<option value="">Sin modelo seleccionado</option>${options}`;
  ["model-active-select", "upcoming-model-select"].forEach((id) => {
    const select = document.getElementById(id);
    if (!select) return;
    select.innerHTML = selectHtml;
    select.value = state.newModelMode ? "" : activeId || (models[0] || {}).model_id || "";
  });
  const active = state.newModelMode ? {} : models.find((model) => model.model_id === activeId) || models[0] || {};
  renderActiveModel(active);
  document.getElementById("models-list").innerHTML = models.map((model) => `
    <article class="model-row ${model.active ? "active" : ""}">
      <div><strong>${escapeHtml(model.model_name || model.model_id)}</strong><small>${escapeHtml(model.model_id)} - ${escapeHtml(model.model_label || model.model_type || "")}</small></div>
      <span>${escapeHtml(modelMarketLabel(model))}</span>
      <b>${model.active ? "Activo" : ""}</b>
    </article>`).join("") || loadingHtml("Entrena tu primer modelo Mundial");
}

function renderActiveModel(model) {
  document.getElementById("active-model-state").innerHTML = [
    predictionCard("Activo", model && model.trained ? (model.model_name || model.model_id) : "Sin modelo"),
    predictionCard("Tipo", (model && (model.model_label || model.model_type)) || "-"),
    predictionCard("Mercados", modelMarketLabel(model || {})),
    predictionCard("Eval", evalStrategyLabel(model && model.eval_strategy, model)),
  ].join("");
}

function syncModelSelects(event) {
  const value = event && event.target ? event.target.value : selectedModelId();
  state.newModelMode = !value;
  ["model-active-select", "upcoming-model-select"].forEach((id) => {
    const select = document.getElementById(id);
    if (select) select.value = value || "";
  });
  if (value) renderActiveModel(state.models.find((model) => model.model_id === value) || {});
}

function selectedModelId() {
  if (state.newModelMode) return "";
  const activeSelect = document.getElementById("model-active-select");
  const upcomingSelect = document.getElementById("upcoming-model-select");
  return (activeSelect && activeSelect.value) || (upcomingSelect && upcomingSelect.value) || state.activeModelId || "";
}

function startNewWorldcupModel() {
  clearAlert();
  state.newModelMode = true;
  ["model-active-select", "upcoming-model-select"].forEach((id) => {
    const select = document.getElementById(id);
    if (select) select.value = "";
  });
  document.querySelectorAll(".model-row.active").forEach((row) => row.classList.remove("active"));
  const modelType = document.getElementById("worldcup-model-type").value || ((state.trainingOptions || {}).defaults || {}).model_type || "xgboost";
  state.trainingControlsApplied = false;
  renderTrainingControls(state.trainingOptions, {});
  applyModelDefaults(modelType, true);
  const modelId = document.getElementById("worldcup-model-id");
  modelId.value = nextWorldcupModelId(modelType);
  modelId.placeholder = modelId.value;
  modelId.dataset.autofilled = "true";
  document.getElementById("worldcup-tuning-enabled").checked = false;
  applyTuningLocks();
  const simMlToggle = document.getElementById("sim-use-ml-model");
  if (simMlToggle) simMlToggle.checked = false;
  document.getElementById("upcoming-summary").textContent = "Nuevo modelo pendiente de entrenamiento";
  document.getElementById("upcoming-predictions").innerHTML = loadingHtml("Sin modelo seleccionado");
  document.getElementById("upcoming-predictions-table").innerHTML = "";
  document.getElementById("training-model-state").innerHTML = [
    predictionCard("Modo", "Nuevo modelo"),
    predictionCard("Modelo", "Sin guardar"),
    predictionCard("Mercados", "1X2 + U/O 0.5-3.5"),
    predictionCard("Eval", "pendiente"),
  ].join("");
  document.getElementById("training-metric-cards").innerHTML = loadingHtml("Entrena el nuevo modelo");
  document.getElementById("training-confusion-matrix").innerHTML = loadingHtml("Matriz pendiente");
  document.getElementById("training-tuning-flow").innerHTML = tuningFlowHtml({ enabled: false });
  document.getElementById("training-features").innerHTML = loadingHtml("Features pendientes");
  document.getElementById("training-model-params").innerHTML = loadingHtml("Parametros pendientes");
  document.getElementById("simulation-summary").textContent = `Nuevo modelo preparado: ${modelId.value}`;
  renderActiveModel({});
}

async function loadSelectedModel() {
  clearAlert();
  const modelId = selectedModelId();
  if (!modelId) {
    showError("Entrena o selecciona un modelo Mundial primero.");
    return;
  }
  try {
    state.newModelMode = false;
    const result = await api("/api/mundial/models/select", jsonOptions({ model_id: modelId }));
    state.activeModelId = result.active_model_id || (result.selected || {}).model_id || modelId;
    state.trainingControlsApplied = false;
    renderModelsCatalog(result);
    renderTrainingControls(state.trainingOptions, result.selected || {});
    const modelIdInput = document.getElementById("worldcup-model-id");
    if (modelIdInput && (result.selected || {}).model_id) {
      modelIdInput.value = result.selected.model_id;
      modelIdInput.dataset.autofilled = "true";
    }
    renderModelState(result.selected || {}, state.training || {});
    renderTrainingVisuals(result.selected || {}, state.training || {});
    renderTrainingTables(result.selected || {}, state.training || {});
    const simMlToggle = document.getElementById("sim-use-ml-model");
    if (simMlToggle) simMlToggle.checked = true;
    document.getElementById("simulation-summary").textContent = `Híbrido activo: ${result.selected.model_name || result.selected.model_id}`;
  } catch (error) {
    showError(error.message);
  }
}

async function deleteSelectedModel() {
  clearAlert();
  const modelId = selectedModelId();
  if (!modelId) {
    showError("Selecciona un modelo para borrar.");
    return;
  }
  try {
    const result = await api(`/api/mundial/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
    renderModelsCatalog(result);
    document.getElementById("simulation-summary").textContent = result.models && result.models.length
      ? `Modelos restantes: ${result.models.length}`
      : "Sin modelos híbridos cargados.";
  } catch (error) {
    showError(error.message);
  }
}

async function clearWorldcupMaintenance() {
  clearAlert();
  document.getElementById("simulation-summary").textContent = "Limpiando modelos y cache Mundial...";
  try {
    const result = await api("/api/mundial/maintenance/clear", jsonOptions({ clear_cache: true }));
    state.models = (result.models || {}).models || [];
    state.activeModelId = (result.models || {}).active_model_id || "";
    state.training = result.training || state.training;
    renderModelsCatalog(result.models || {});
    renderTrainingStatus(result.training || state.training || {});
    document.getElementById("simulation-summary").textContent = `Limpieza completa: ${((result.removed || []).length)} rutas procesadas.`;
  } catch (error) {
    showError(error.message);
  }
}

async function loadPlayers(refresh) {
  try {
    const result = await api(`/api/mundial/players?refresh=${refresh ? "true" : "false"}`);
    state.players = result.players || [];
    renderPlayers(result);
  } catch (error) {
    showError(error.message);
  }
}

function renderPlayers(payload) {
  document.getElementById("players-source").textContent = payload.source || "";
  const players = payload.players || [];
  document.getElementById("players-list").innerHTML = players.slice(0, 90).map((player) => `
    <article class="player-row">
      ${playerPhotoHtml(player)}
      <div><strong>${escapeHtml(player.name)}</strong><small>${escapeHtml(player.team.name)} - ${escapeHtml(player.position || "Posicion pendiente")}</small></div>
      ${flagHtml(player.team)}
    </article>`).join("") || loadingHtml("Jugadores pendientes");
  renderTable("players-table", payload.table);
}

async function loadTrainingStatus() {
  try {
    const result = await api("/api/mundial/training/status");
    state.training = result;
    state.trainingOptions = result.options || state.trainingOptions;
    renderTrainingStatus(result);
  } catch (error) {
    showError(error.message);
  }
}

async function downloadTrainingDataset() {
  clearAlert();
  document.getElementById("training-status").textContent = "Descargando Kaggle + All matches...";
  try {
    const result = await api("/api/mundial/training/download-kaggle", jsonOptions({ force: false }));
    const refreshed = await api("/api/mundial/training/status");
    state.training = {
      ...refreshed,
      international_recent: result.international_recent || refreshed.international_recent,
      download_warnings: trainingStatusWarnings(result),
    };
    state.trainingOptions = refreshed.options || state.trainingOptions;
    renderTrainingStatus(state.training);
  } catch (error) {
    showError(error.message);
    await loadTrainingStatus();
  }
}

async function prepareTrainingEtl() {
  clearAlert();
  document.getElementById("training-status").textContent = "Preparando ETL...";
  try {
    const result = await api("/api/mundial/training/prepare-etl", jsonOptions({ force: true }));
    state.training = result;
    renderTrainingStatus(result);
  } catch (error) {
    showError(error.message);
    await loadTrainingStatus();
  }
}

async function refreshPlayerSnapshots() {
  clearAlert();
  document.getElementById("training-status").textContent = "Actualizando snapshots de jugadores...";
  try {
    const result = await api("/api/mundial/training/player-snapshots", jsonOptions({ refresh: true, limit: 8 }));
    const refreshed = await api("/api/mundial/training/status");
    state.training = {
      ...refreshed,
      snapshot_refresh: result,
    };
    state.trainingOptions = refreshed.options || state.trainingOptions;
    renderTrainingStatus(state.training);
  } catch (error) {
    showError(error.message);
    await loadTrainingStatus();
  }
}

async function trainWorldCupModel(walkForwardMode = "none") {
  clearAlert();
  if (!state.training || !state.training.etl_ready) {
    showError("Primero ejecuta Preparar ETL para dejar listo el dataset de entrenamiento.");
    return;
  }
  if (state.training.etl_stale) {
    showError("El ETL esta desactualizado; vuelve a ejecutar Preparar ETL antes de entrenar.");
    return;
  }
  const modelId = ensureWorldcupModelId();
  if (!modelId) {
    showError("No se pudo generar el nombre del nuevo modelo.");
    return;
  }
  const modeLabel = walkForwardMode === "result_only"
      ? "Reentrenando con partido nuevo..."
      : "Entrenando híbrido Mundial...";
  document.getElementById("training-status").textContent = modeLabel;
  try {
    const job = await api("/api/mundial/models/train", jsonOptions(trainingPayload(walkForwardMode)));
    trackWorldcupJob(job, "training");
    document.getElementById("simulation-summary").textContent = "Entrenamiento en proceso...";
  } catch (error) {
    showError(error.message);
  }
}

function renderTrainingStatus(payload) {
  const model = payload.model || {};
  state.trainingOptions = payload.options || state.trainingOptions;
  renderTrainingControls(state.trainingOptions, model);
  const hardware = (model.hardware && model.trained) ? model.hardware : ((state.trainingOptions || {}).hardware || {});
  renderHeroHardware(hardware);
  const sourceAvailable = payload.available || payload.etl_ready || Boolean((payload.international_recent || {}).available);
  document.getElementById("training-status").textContent = sourceAvailable
    ? `${payload.train_rows || 0} train listo - ${etlStatusLabel(payload)} - ${evalStrategyLabel(payload.eval_strategy, payload)}`
    : "all_matches.csv no disponible";
  document.getElementById("training-source").textContent = `all_matches.csv - ${payload.training_mode || "sin modo"} - ${payload.prepared_label_source || "fuente pendiente"}`;
  document.getElementById("training-summary").innerHTML = datasetSummaryHtml(payload);
  const trainDisabled = !payload.etl_ready || payload.etl_stale;
  document.getElementById("training-train").disabled = trainDisabled;
  document.getElementById("training-retrain-base").disabled = trainDisabled;
  renderWalkForwardNotice(payload.walk_forward_refresh || {});
  renderModelState(model, payload);
  renderTable("training-preview", payload.preview);
  renderTrainingWarnings(trainingStatusWarnings(payload));
  renderTrainingVisuals(model, payload);
  renderTrainingTables(model, payload);
}

function renderTrainingResult(payload) {
  renderHeroHardware(payload.hardware || {});
  renderTrainingWarnings(payload.warnings || []);
  renderWalkForwardNotice(((state.training || {}).walk_forward_refresh) || {});
  document.getElementById("training-summary").innerHTML = datasetSummaryHtml({
    ...(state.training || {}),
    train_rows: payload.train_rows,
    eval_rows: payload.eval_rows,
    eval_strategy: payload.eval_strategy,
    prediction_rows: payload.prediction_rows,
    target_column: (payload.model || {}).target_column || payload.target_column || payload.effective_target,
  });
  renderModelState(payload.model || {}, payload);
  renderTrainingVisuals(payload.model || {}, payload);
  renderTrainingTables(payload.model || {}, payload);
}

function renderTrainingVisuals(model, payload) {
  const markets = trainingMarketSections(model, payload);
  renderEtlFlow((model.etl_steps || payload.etl_steps || []));
  renderMetricCards(markets);
  renderConfusionMatrix(markets);
  renderTuningFlow(markets);
  renderFeatureList(markets, model || payload || {});
}

function trainingMarketSections(model, payload) {
  const markets = (model && model.markets) || (payload && payload.markets) || {};
  const keys = trainingMarketOrder.filter((key) => markets[key]);
  if (keys.length) {
    return keys.map((key) => ({ key, label: markets[key].label || marketLabel(key), ...markets[key] }));
  }
  const target = (model && (model.effective_target || model.requested_target)) || (payload && (payload.effective_target || payload.requested_target)) || "result";
  return [{
    key: target && String(target).startsWith("over_under_") ? target : target === "goals_distribution" ? "goals_distribution" : "result",
    label: marketLabel(target),
    metrics: (model && model.metrics) || (payload && payload.metrics) || {},
    confusion_matrix: (model && model.confusion_matrix) || (payload && payload.confusion_matrix) || {},
    tuning_trace: (model && model.tuning_trace) || (payload && payload.tuning_trace) || (model && model.tuning) || {},
    top_features: (model && model.top_features) || [],
    train_rows: payload && payload.train_rows,
    eval_rows: payload && payload.eval_rows,
  }];
}

function datasetSummaryHtml(payload) {
  const targetYear = targetWorldcupYear(payload);
  const last30Eval = payload.eval_strategy === "last_30_international_test";
  const evalValue = last30Eval
    ? `${payload.test_rows || 30} partidos`
    : payload.test_rows
    ? `${payload.test_rows} filas test`
    : `${payload.eval_rows || 0} holdout`;
  const walkForward = payload.walk_forward || {};
  const refresh = payload.walk_forward_refresh || {};
  const international = payload.international_recent || {};
  const internationalReady = Boolean(international.available);
  return [
    datasetCard("Objetivo", `Mundial ${targetYear}`, "torneo operativo"),
    datasetCard("Labels", "Internacionales no Mundial", payload.prepared_label_source || "all_matches.csv"),
    datasetCard("Archivos", (payload.files || []).length, "CSV/XLS detectados"),
    datasetCard("ETL", etlStatusShort(payload), payload.prepared_label_source || "preparar artifact"),
    datasetCard("Train etiquetado", payload.train_rows || 0, payload.training_mode || "sin modo"),
    datasetCard("Eval", evalValue, evalStrategyLabel(payload.eval_strategy, payload)),
    datasetCard(`Predicción ${targetYear}`, payload.prediction_rows || 0, "filas sin label usadas como features"),
    datasetCard("Features equipo", payload.team_feature_rows || 0, "equipos disponibles"),
    datasetCard("All matches", internationalReady ? international.rows || 0 : "faltante", internationalStatusDetail(international)),
    datasetCard("Walk-forward", refresh.completed_results || walkForward.completed_results || 0, `${refresh.ready_result_only || 0} resultado listo · ${refresh.needs_player_snapshot || 0} snapshot pendiente`),
    datasetCard("Target", targetDisplayLabel(payload.target_column || "-"), "label entrenable"),
  ].join("");
}

function internationalStatusDetail(international) {
  const status = international || {};
  if (status.available) {
    const source = status.source_path && status.source_path !== status.file_path ? ` · ${status.source_path}` : "";
    return `ultimos 15 activos${source}`;
  }
  const reason = status.reason || "all_matches.csv no disponible";
  const path = status.file_path || "storage/worldcup/international/all_matches.csv";
  return `${reason} · ${path}`;
}

function trainingStatusWarnings(payload) {
  const international = (payload && payload.international_recent) || {};
  const snapshotWarning = (payload && payload.snapshot_refresh && payload.snapshot_refresh.warning) || "";
  const warnings = [
    ...((payload && payload.prepared_warnings) || []),
    ...((payload && payload.market_warnings) || []),
    ...((payload && payload.api_football_warnings) || []),
    ...((payload && payload.download_warnings) || []),
    international.warning || "",
    snapshotWarning,
    !international.available && international.reason ? `All matches: ${international.reason}` : "",
  ].filter(Boolean).map((item) => String(item));
  return [...new Set(warnings)];
}

function targetDisplayLabel(value) {
  const text = String(value || "");
  if (text.includes("GoalsDistribution") || text.includes("OverUnder")) return "Label 1X2";
  return text || "-";
}

function etlStatusLabel(payload) {
  if (!payload || !payload.etl_ready) return "ETL pendiente";
  return payload.etl_stale ? "ETL desactualizado" : "ETL listo";
}

function etlStatusShort(payload) {
  if (!payload || !payload.etl_ready) return "Pendiente";
  return payload.etl_stale ? "Desactualizado" : "Listo";
}

function datasetCard(label, value, detail) {
  return `<article class="dataset-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail || "")}</small></article>`;
}

function renderModelState(model, payload) {
  document.getElementById("training-model-state").innerHTML = [
    predictionCard("Modelo", model.trained ? (model.model_label || payload.model_type || "Listo") : "Pendiente"),
    predictionCard("Mercados", modelMarketLabel(model.trained ? model : payload)),
    predictionCard("Eval", evalStrategyLabel(model.eval_strategy || payload.eval_strategy, model.trained ? model : payload)),
    predictionCard("Walk-forward", walkForwardModeLabel((model.walk_forward_mode || (model.walk_forward_summary || {}).mode || "none"))),
  ].join("");
}

function modelMarketLabel(model) {
  if (!model) return "-";
  if (model.bundle || model.market_mode === "dual_markets" || model.requested_target === "dual_markets") {
    return "1X2 + U/O 0.5-3.5";
  }
  const target = model.effective_target || model.requested_target || model.training_target || "";
  const line = goalMarketLines.find((item) => item.key === target);
  if (line) return line.label;
  if (target === "goals_distribution") return "Distribución goles";
  if (target === "team_strength") return "1X2 team-strength";
  if (target === "result") return "1X2";
  return target || "-";
}

function walkForwardModeLabel(mode) {
  if (mode === "result_only") return "Partido base";
  return "Sin incremental";
}

function marketLabel(key) {
  const line = goalMarketLines.find((item) => item.key === key);
  if (line) return line.label;
  if (key === "goals_distribution") return "Distribución goles";
  if (key === "team_strength") return "1X2 team-strength";
  return "1X2";
}

function evalStrategyLabel(strategy, payload) {
  if (strategy === "last_30_international_test") return "ultimos 30 internacionales";
  if (strategy === "final_worldcup_test") {
    const year = String((payload && payload.final_test_year) || "").trim();
    return year ? `Mundial ${year}` : "Mundial historico";
  }
  if (strategy === "test_file") return "test etiquetado";
  if (strategy === "holdout_temporal") return "holdout temporal";
  if (strategy === "holdout_from_train") return "holdout desde train";
  if (strategy === "unavailable") return "sin evaluacion";
  return strategy || "pendiente";
}

function targetWorldcupYear(payload) {
  return String((payload && payload.target_worldcup_year) || "2026").trim();
}

function renderTrainingControls(options, model) {
  if (!options) return;
  const models = options.models || [];
  const modelSelect = document.getElementById("worldcup-model-type");
  if (!modelSelect.options.length) {
    modelSelect.innerHTML = models.map((item) => `<option value="${escapeAttr(item.key)}">${escapeHtml(item.label)}</option>`).join("");
  }
  const selectedModel = model.model_type || (options.defaults || {}).model_type || modelSelect.value || "xgboost";
  modelSelect.value = selectedModel;
  const modelId = model.model_id || autoWorldcupModelId(selectedModel);
  const modelIdInput = document.getElementById("worldcup-model-id");
  if (modelIdInput && (!modelIdInput.value || modelIdInput.dataset.autofilled !== "false")) {
    modelIdInput.value = modelId;
    modelIdInput.dataset.autofilled = "true";
  }
  if (!state.trainingControlsApplied) {
    document.getElementById("worldcup-device").value = preferredTrainingDevice(options, model);
    document.getElementById("worldcup-n-jobs").value = (options.defaults || {}).n_jobs ?? -1;
    document.getElementById("worldcup-n-trials").value = (options.defaults || {}).n_trials || 12;
    document.getElementById("worldcup-objective").value = (options.defaults || {}).objective || "F1";
    document.getElementById("worldcup-optuna-sampler").value = (options.defaults || {}).optuna_sampler || "tpe";
    document.getElementById("worldcup-optuna-pruner").value = (options.defaults || {}).optuna_pruner || "none";
    document.getElementById("worldcup-tune-params").value = (options.defaults || {}).tune_params || "all";
    applyModelDefaults(selectedModel, false);
    state.trainingControlsApplied = true;
  }
  applyTuningLocks();
}

function applyModelDefaults(modelKey, force) {
  const models = ((state.trainingOptions || {}).models || []);
  const model = models.find((item) => item.key === modelKey) || {};
  const defaults = model.defaults || {};
  const mapping = {
    n_estimators: "worldcup-n-estimators",
    learning_rate: "worldcup-learning-rate",
    max_depth: "worldcup-max-depth",
    min_child_weight: "worldcup-min-child-weight",
    lambda_regularization: "worldcup-lambda-regularization",
    alpha_regularization: "worldcup-alpha-regularization",
    num_leaves: "worldcup-num-leaves",
    min_child_samples: "worldcup-min-child-samples",
    minibatch_frac: "worldcup-minibatch-frac",
    l2_leaf_reg: "worldcup-l2-leaf-reg",
    random_strength: "worldcup-random-strength",
  };
  Object.entries(mapping).forEach(([key, id]) => {
    const input = document.getElementById(id);
    if (!input) return;
    if (defaults[key] === undefined) {
      input.value = "";
      input.disabled = true;
      input.dataset.unavailable = "true";
      return;
    }
    input.dataset.unavailable = "false";
    input.disabled = false;
    if (force || input.value === "") input.value = defaults[key];
  });
  const natural = document.getElementById("worldcup-natural-gradient");
  natural.disabled = defaults.natural_gradient === undefined;
  natural.dataset.unavailable = defaults.natural_gradient === undefined ? "true" : "false";
  natural.checked = Boolean(defaults.natural_gradient);
  const modelIdInput = document.getElementById("worldcup-model-id");
  if (modelIdInput && (force || !modelIdInput.value || modelIdInput.dataset.autofilled !== "false")) {
    modelIdInput.value = state.newModelMode ? nextWorldcupModelId(modelKey) : autoWorldcupModelId(modelKey);
    modelIdInput.dataset.autofilled = "true";
  }
  applyTuningLocks();
}

function preferredTrainingDevice(options, model) {
  const defaults = options.defaults || {};
  const hardware = options.hardware || {};
  const requested = (model.hardware || {}).requested_device || "";
  const fallback = defaults.device || "auto";
  if (requested === "cpu" && hardware.cuda_available) return fallback;
  if (requested === "cuda" && !hardware.cuda_available) return fallback;
  return requested || fallback;
}

function autoWorldcupModelId(modelKey) {
  const shortModel = { xgboost: "xgb", lightgbm: "lgbm", catboost: "cat", ngboost: "ngb" }[modelKey] || modelKey || "model";
  return `mundial-${shortModel}-hibrido`;
}

function nextWorldcupModelId(modelKey) {
  const base = autoWorldcupModelId(modelKey);
  const existing = new Set((state.models || []).map((model) => String(model.model_id || "")));
  if (!existing.has(base)) return base;
  for (let index = 2; index < 1000; index += 1) {
    const candidate = `${base}-${index}`;
    if (!existing.has(candidate)) return candidate;
  }
  return `${base}-${Date.now()}`;
}

function ensureWorldcupModelId() {
  const input = document.getElementById("worldcup-model-id");
  if (!input) return "";
  const current = input.value.trim();
  if (current) return current;
  const modelType = document.getElementById("worldcup-model-type").value || ((state.trainingOptions || {}).defaults || {}).model_type || "xgboost";
  input.value = nextWorldcupModelId(modelType);
  input.dataset.autofilled = "true";
  return input.value;
}

function renderTrainingWarnings(warnings) {
  document.getElementById("training-warnings").innerHTML = (warnings || []).map((warning) => `<span>${escapeHtml(warning)}</span>`).join("");
}

function renderWalkForwardNotice(refresh) {
  const container = document.getElementById("training-walkforward-notice");
  if (!container) return;
  const items = [];
  if (refresh.ready_result_only) items.push(`Resultado listo para reentreno base: ${refresh.ready_result_only}`);
  if (refresh.needs_player_snapshot) items.push(`Snapshot de jugadores pendiente: ${refresh.needs_player_snapshot}`);
  if (refresh.pending_results) items.push(`Resultados pendientes: ${refresh.pending_results}`);
  if (refresh.latest_played_fixture) items.push(`Último jugado: ${refresh.latest_played_fixture}`);
  if (refresh.note && !items.includes(refresh.note)) items.push(refresh.note);
  container.innerHTML = items.map((item) => `<span>${escapeHtml(item)}</span>`).join("") || `<span>Sin alertas de walk-forward.</span>`;
}

function renderEtlFlow(steps) {
  document.getElementById("training-etl-flow").innerHTML = (steps || []).map((step, index) => `
    <article class="etl-step ${escapeAttr(step.status || "info")}">
      <span>${escapeHtml(index + 1)}</span>
      <div><strong>${escapeHtml(step.name || "")}</strong><small>${escapeHtml(step.detail || "")}</small></div>
      <b>${escapeHtml(step.count ?? "")}</b>
    </article>`).join("") || loadingHtml("ETL pendiente");
}

function renderMetricCards(markets) {
  const sections = Array.isArray(markets) ? markets : [{ label: "Evaluacion", metrics: markets || {} }];
  document.getElementById("training-metric-cards").innerHTML = sections.map((market) => {
    const evalMetrics = (market.metrics && (market.metrics.eval || market.metrics.Eval)) || {};
    const rows = ["Accuracy", "F1", "Precision", "Recall"].map((key) => predictionCard(key, evalMetrics[key] ?? "-")).join("");
    return `<section class="market-panel"><header><strong>${escapeHtml(market.label || "Mercado")}</strong><small>${escapeHtml((market.train_rows ?? "-"))} train / ${escapeHtml((market.eval_rows ?? "-"))} eval</small></header><div class="market-card-grid">${rows}</div></section>`;
  }).join("");
}

function renderConfusionMatrix(payload) {
  if (Array.isArray(payload)) {
    document.getElementById("training-confusion-matrix").innerHTML = payload.map((market) => `
      <section class="market-panel confusion-panel">
        <header>
          <strong>${escapeHtml(market.label || "Mercado")}</strong>
          <small>${escapeHtml(confusionTargetLabel(market.effective_target || market.key || ""))} - FP/FN por clase</small>
        </header>
        ${confusionMatrixHtml(market.confusion_matrix || {})}
      </section>`).join("");
    return;
  }
  document.getElementById("training-confusion-matrix").innerHTML = confusionMatrixHtml(payload || {});
}

function confusionMatrixHtml(payload) {
  const labels = payload.labels || [];
  const matrix = payload.matrix || [];
  if (!labels.length || !matrix.length) {
    return loadingHtml("Matriz pendiente");
  }
  const maxValue = Math.max(...matrix.flat().map((value) => Number(value) || 0), 1);
  const header = `<div class="confusion-axis">Actual \\ Predicho</div>${labels.map((label) => `<strong>${escapeHtml(label)}</strong>`).join("")}`;
  const rows = matrix.map((row, rowIndex) => `
    <strong>${escapeHtml(labels[rowIndex])}</strong>
    ${row.map((value, colIndex) => {
      const intensity = Math.max(0.12, Number(value || 0) / maxValue);
      const correct = rowIndex === colIndex ? " correct" : "";
      return `<span class="confusion-cell${correct}" style="--intensity:${escapeAttr(intensity)}"><b>${escapeHtml(value)}</b></span>`;
    }).join("")}
  `).join("");
  return `
    <div class="confusion-grid" style="grid-template-columns: 130px repeat(${labels.length}, minmax(82px, 1fr))">${header}${rows}</div>
    ${confusionSummaryHtml(labels, matrix)}`;
}

function confusionSummaryHtml(labels, matrix) {
  const totals = labels.map((_, index) => ({
    label: labels[index],
    tp: Number((matrix[index] || [])[index] || 0),
    fp: matrix.reduce((sum, row, rowIndex) => sum + (rowIndex === index ? 0 : Number(row[index] || 0)), 0),
    fn: (matrix[index] || []).reduce((sum, value, colIndex) => sum + (colIndex === index ? 0 : Number(value || 0)), 0),
  }));
  return `
    <div class="confusion-summary">
      ${totals.map((item) => `
        <article>
          <span>${escapeHtml(item.label)}</span>
          <strong>FP ${escapeHtml(item.fp)}</strong>
          <small>FN ${escapeHtml(item.fn)} / TP ${escapeHtml(item.tp)}</small>
        </article>`).join("")}
    </div>`;
}

function renderTuningFlow(trace) {
  if (Array.isArray(trace)) {
    document.getElementById("training-tuning-flow").innerHTML = trace.map((market) => `
      <section class="market-panel">
        <header><strong>${escapeHtml(market.label || "Mercado")}</strong><small>${escapeHtml(market.model_id || "")}</small></header>
        ${tuningFlowHtml(market.tuning_trace || market.tuning || {})}
      </section>`).join("");
    return;
  }
  document.getElementById("training-tuning-flow").innerHTML = tuningFlowHtml(trace || {});
}

function tuningFlowHtml(trace) {
  const steps = trace.steps || [];
  const head = trace.enabled
    ? `<div class="tuning-head"><strong>Best ${escapeHtml(trace.objective || "")}: ${escapeHtml(trace.best_value ?? "")}</strong><small>Trial ${escapeHtml(trace.best_trial ?? "")} - ${escapeHtml(trace.trials ?? "")} trials</small></div>`
    : `<div class="tuning-head"><strong>Fine-tuning desactivado</strong><small>Se usaron parametros manuales/default.</small></div>`;
  const items = steps.map((step) => `<article class="tuning-step ${escapeAttr(step.status || "info")}"><strong>${escapeHtml(step.name || "")}</strong><small>${escapeHtml(step.detail || "")}</small></article>`).join("");
  return head + `<div class="tuning-steps">${items}</div>`;
}

function renderFeatureList(markets, model) {
  if (Array.isArray(markets)) {
    const marketHtml = markets.map((market) => `
      <section class="market-panel">
        <header><strong>${escapeHtml(market.label || "Mercado")}</strong><small>${escapeHtml(market.model_id || "")}</small></header>
        ${featureListHtml(market.top_features || [])}
      </section>`).join("");
    document.getElementById("training-features").innerHTML = marketHtml + featureInventoryHtml((model && model.feature_inventory) || {});
    return;
  }
  document.getElementById("training-features").innerHTML = featureListHtml(markets || []) + featureInventoryHtml((model && model.feature_inventory) || {});
}

function featureListHtml(features) {
  return (features || []).slice(0, 10).map((item) => `
    <div class="feature-bar">
      <span>${escapeHtml(item.feature || "")}</span>
      <div><i style="width:${escapeAttr(Math.min(100, Math.max(2, Number(item.importance || 0) * 100)))}%"></i></div>
      <b>${escapeHtml(item.importance ?? "")}</b>
    </div>`).join("") || loadingHtml("Features pendientes");
}

function featureImportanceTable(features) {
  const rows = (features || []).map((item) => ({ Feature: item.feature, Importancia: item.importance }));
  return { columns: rows.length ? ["Feature", "Importancia"] : [], rows, total: rows.length };
}

function featureInventoryHtml(inventory) {
  const features = (inventory && inventory.features) || [];
  if (!features.length) return "";
  const families = (inventory.families || []).map((item) => `
    <span>${escapeHtml(item.family || "")}<b>${escapeHtml(item.count ?? 0)}</b></span>
  `).join("");
  const rows = features.map((item) => `
    <div class="feature-inventory-row">
      <span>${escapeHtml(item.feature || "")}</span>
      <small>${escapeHtml(item.family || "")} · nz train ${escapeHtml(item.train_non_zero_rate ?? 0)} · var ${escapeHtml(item.train_variance ?? 0)}</small>
    </div>
  `).join("");
  return `
    <section class="market-panel feature-inventory-panel">
      <header><strong>Inventario completo</strong><small>${escapeHtml(inventory.feature_count || features.length)} features</small></header>
      <div class="feature-family-list">${families}</div>
      <div class="feature-inventory-list">${rows}</div>
    </section>
  `;
}

function paramsTable(model) {
  const params = (model && model.model_params) || {};
  const tuning = (model && model.tuning) || {};
  const rows = Object.entries(params).map(([key, value]) => ({ Parametro: key, Valor: value }));
  if (tuning.enabled) {
    rows.push({ Parametro: "tuning.best_value", Valor: tuning.best_value ?? "" });
    rows.push({ Parametro: "tuning.best_trial", Valor: tuning.best_trial ?? "" });
  }
  return { columns: rows.length ? ["Parametro", "Valor"] : [], rows, total: rows.length };
}

function renderTrainingTables(model, payload) {
  const markets = trainingMarketSections(model, payload);
  document.getElementById("training-metrics").innerHTML = markets.map((market) => `
    <section class="market-panel">
      <header><strong>${escapeHtml(market.label || "Mercado")}</strong><small>${escapeHtml(confusionTargetLabel(market.effective_target || market.key || ""))}</small></header>
      ${tableHtml(metricsTableFromMarket(market))}
    </section>`).join("");
  document.getElementById("training-model-params").innerHTML = markets.map((market) => `
    <section class="market-panel">
      <header><strong>${escapeHtml(market.label || "Mercado")}</strong><small>${escapeHtml(market.model_name || market.model_id || "")}</small></header>
      ${tableHtml(paramsTable(market))}
    </section>`).join("");
}

function metricsTableFromMarket(market) {
  const metrics = market.metrics || {};
  const rows = Object.entries(metrics).map(([split, values]) => ({ Split: split, ...(values || {}) }));
  const columns = rows.length ? Object.keys(rows[0]) : [];
  return { columns, rows, total: rows.length };
}

function confusionTargetLabel(target) {
  if (String(target || "").startsWith("over_under_")) return "2 clases: Under / Over";
  if (target === "goals_distribution") return "Buckets de goles totales";
  return "3 clases: 1 / X / 2";
}

function parseTuneParams(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw || raw === "all") return ["all"];
  return raw.split(",").map((item) => item.trim()).filter(Boolean);
}

function applyTuningLocks() {
  const tuningEnabled = document.getElementById("worldcup-tuning-enabled").checked;
  const tuneParams = parseTuneParams(document.getElementById("worldcup-tune-params").value);
  const modelKey = document.getElementById("worldcup-model-type").value || "xgboost";
  const models = ((state.trainingOptions || {}).models || []);
  const model = models.find((item) => item.key === modelKey) || {};
  const tunables = model.tunables || {};
  const allSelected = tuneParams.includes("all");
  const controlled = allSelected ? Object.keys(tunables) : tuneParams;
  const lockedInputs = {
    n_estimators: "worldcup-n-estimators",
    learning_rate: "worldcup-learning-rate",
    max_depth: "worldcup-max-depth",
    min_child_weight: "worldcup-min-child-weight",
    lambda_regularization: "worldcup-lambda-regularization",
    alpha_regularization: "worldcup-alpha-regularization",
    num_leaves: "worldcup-num-leaves",
    min_child_samples: "worldcup-min-child-samples",
    minibatch_frac: "worldcup-minibatch-frac",
    l2_leaf_reg: "worldcup-l2-leaf-reg",
    random_strength: "worldcup-random-strength",
    natural_gradient: "worldcup-natural-gradient",
  };
  Object.entries(lockedInputs).forEach(([key, id]) => {
    const input = document.getElementById(id);
    if (!input) return;
    const unavailable = input.dataset.unavailable === "true";
    const shouldLock = tuningEnabled && controlled.includes(key);
    if (shouldLock) {
      input.dataset.lockedByTuning = "true";
      input.disabled = true;
    } else if (input.dataset.lockedByTuning === "true") {
      input.dataset.lockedByTuning = "false";
      if (!unavailable) input.disabled = false;
    }
  });
  ["training-manual-params", "training-advanced-params"].forEach((id) => {
    const block = document.getElementById(id);
    if (!block) return;
    const hasLockedField = [...block.querySelectorAll("input,select")].some((node) => node.dataset.lockedByTuning === "true");
    block.classList.toggle("tuning-locked", hasLockedField);
  });
  const note = document.getElementById("training-tuning-lock-status");
  if (!note) return;
  if (!tuningEnabled) {
    note.innerHTML = `<span>Optuna desactivado. Se usan los parámetros visibles del formulario.</span>`;
    return;
  }
  const detail = allSelected
    ? "Optuna controla todos los hiperparámetros tunables, incluyendo los avanzados."
    : `Optuna controla: ${controlled.join(", ") || "ninguno"}.`;
  note.innerHTML = `<span>${escapeHtml(detail)}</span>`;
}

async function runUpcomingPredictions() {
  clearAlert();
  const limit = Number(document.getElementById("upcoming-predict-limit").value || 8);
  const group = document.getElementById("upcoming-group-filter").value || "";
  const modelId = document.getElementById("upcoming-model-select").value || selectedModelId();
  const pipelineMode = document.getElementById("upcoming-pipeline-mode").value || "default_ai_poisson";
  document.getElementById("upcoming-summary").textContent = `Generando reporte con Poisson ultimos ${currentPoissonRecentMatches()}...`;
  try {
    const job = await api("/api/mundial/predict-upcoming-report", jsonOptions({
      ...simulationPayload({
        model_id: modelId,
        use_ml_model: pipelineMode === "poisson_sota" ? false : Boolean(document.getElementById("sim-use-ml-model").checked),
        score_model: "independent_poisson",
      }),
      pipeline_mode: pipelineMode,
      limit,
      group,
      bayes_profile: (document.getElementById("upcoming-bayes-profile") || {}).value || "deep",
      sota_device: (document.getElementById("upcoming-sota-device") || {}).value || "auto",
    }));
    trackWorldcupJob(job, "upcoming-report");
  } catch (error) {
    document.getElementById("upcoming-report").innerHTML = loadingHtml("Reporte no disponible");
    showError(error.message);
  }
}

function syncUpcomingPipelineControls() {
  const mode = (document.getElementById("upcoming-pipeline-mode") || {}).value || "default_ai_poisson";
  const isSota = mode === "poisson_sota";
  const defaultControls = document.getElementById("upcoming-default-controls");
  const sotaControls = document.getElementById("upcoming-sota-controls");
  if (defaultControls) defaultControls.classList.toggle("hidden", isSota);
  if (sotaControls) sotaControls.classList.toggle("hidden", !isSota);
  const mlToggle = document.getElementById("sim-use-ml-model");
  if (mlToggle && isSota) mlToggle.checked = false;
}

function renderUpcomingReport(report) {
  state.lastUpcomingReport = report;
  const summary = report.summary || {};
  const fixtures = report.fixture_reports || [];
  const hardware = summary.hardware || {};
  const warnings = summary.warnings || [];
  document.getElementById("upcoming-summary").textContent =
    `${summary.pipeline_label || "Reporte"} - ${summary.returned || 0}/${summary.requested || 0} partidos - ${summary.group || "Todos"} - Poisson ultimos ${summary.poisson_recent_matches || currentPoissonRecentMatches()} - ${summary.report_id || report.report_id || ""}`;
  document.getElementById("upcoming-predictions").innerHTML = "";
  document.getElementById("upcoming-report").innerHTML = `
    <div class="report-summary-grid">
      ${reportSummaryCard("Pipeline", summary.pipeline_label || summary.pipeline_mode || "-")}
      ${reportSummaryCard("Fuerza global", globalConsensusStrength(fixtures))}
      ${reportSummaryCard("Partidos", `${summary.returned || 0}/${summary.requested || 0}`)}
      ${reportSummaryCard("Hardware", `${hardware.actual_device || "cpu"} · ${hardware.requested_device || "auto"}`)}
      ${reportSummaryCard("Guardado", report.report_path || "latest.json")}
    </div>
    ${warnings.length ? `<div class="warning-list">${warnings.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
    <div class="upcoming-grid">
      ${fixtures.map((fixtureReport) => reportFixtureCardHtml(fixtureReport)).join("") || loadingHtml("Sin fixtures futuros")}
    </div>`;
  renderTable("upcoming-predictions-table", report.table);
}

function reportSummaryCard(label, value) {
  return `<article class="report-summary-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>`;
}

function globalConsensusStrength(fixtures) {
  const counts = new Map();
  (fixtures || []).forEach((item) => {
    const strength = ((item.consensus || {}).strength) || "Baja";
    counts.set(strength, (counts.get(strength) || 0) + 1);
  });
  if (!counts.size) return "-";
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

function reportFixtureCardHtml(report) {
  const fixture = report.fixture || {};
  const consensus = report.consensus || {};
  const models = report.models || [];
  const topModels = report.top_models_1x2 || [];
  const stats = report.model_statistics || {};
  const scoreDistribution = report.consensus_score_distribution || {};
  const homeAsset = fixture.home_asset || assetFor(fixture.home || "");
  const awayAsset = fixture.away_asset || assetFor(fixture.away || "");
  const consensusClass = ["Baja", ""].includes(consensus.strength || "") ? "low" : "";
  const warnings = report.warnings || [];
  return `<article class="upcoming-card report-fixture-card">
    <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || "")}</strong></header>
    <div class="upcoming-match">
      <div class="upcoming-team">${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
      <span>vs</span>
      <div class="upcoming-team away">${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
    </div>
    <div class="prediction-pick">
      <span>Consenso · ${escapeHtml(consensus.eligible_models || 0)} modelos válidos</span>
      <strong>${escapeHtml(consensus.outcome_label || "-")} · ${escapeHtml(consensus.strength || "Baja")}</strong>
    </div>
    <span class="consensus-badge ${escapeAttr(consensusClass)}">${escapeHtml(Math.round(Number(consensus.outcome_share || 0) * 100))}% 1X2 · ${escapeHtml(Math.round(Number(consensus.signature_share || 0) * 100))}% firma</span>
    ${reportTopModelsHtml(topModels)}
    ${reportOutcomeStatsHtml(stats, consensus, fixture)}
    ${reportConsensusScoreHtml(scoreDistribution)}
    ${reportTotalsStatsHtml(stats, consensus)}
    ${allModelsDetailsHtml(models)}
    ${warnings.length ? `<div class="warning-list compact">${warnings.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
  </article>`;
}

function reportTopModelsHtml(topModels) {
  const models = topModels || [];
  if (!models.length) return "";
  return `<section class="report-panel top-models-panel">
    <header><strong>Top 4 modelos 1/X/2</strong><small>Ordenados por confianza del pick</small></header>
    <div class="top-models-grid">
      ${models.map((model) => {
        const expected = model.expected_goals || {};
        return `<div class="top-model-row">
          <b>#${escapeHtml(model.rank || "")}</b>
          <strong>${escapeHtml(model.model_label || model.model_key || "")}</strong>
          <span>${escapeHtml(model.pick_label || "-")} · ${escapeHtml(model.team || "")}</span>
          <em>${escapeHtml(formatNumber(model.confidence ?? 0))}%</em>
          <small>${escapeHtml(model.top_score || "-")} · λ ${escapeHtml(formatNumber(expected.home ?? "-"))}/${escapeHtml(formatNumber(expected.away ?? "-"))}</small>
        </div>`;
      }).join("")}
    </div>
  </section>`;
}

function reportOutcomeStatsHtml(stats, consensus, fixture) {
  const outcomeStats = (stats && stats.outcomes) || {};
  const outcomeCounts = (consensus && consensus.outcome_counts) || {};
  const eligible = Math.max(Number((stats && stats.model_count) || (consensus && consensus.eligible_models) || 0), 1);
  const outcomes = [
    { key: "home", label: "1", team: fixture.home || "Local" },
    { key: "draw", label: "X", team: "Empate" },
    { key: "away", label: "2", team: fixture.away || "Visitante" },
  ];
  return `<section class="report-panel">
    <header><strong>Distribución 1/X/2</strong><small>Promedio y dispersión entre modelos</small></header>
    <div class="outcome-list stat-outcomes">
      ${outcomes.map((item) => {
        const summary = outcomeStats[item.key] || {};
        const avg = Number.isFinite(Number(summary.avg)) ? Number(summary.avg) : (Number(outcomeCounts[item.key] || 0) / eligible) * 100;
        const active = item.key === ((consensus || {}).outcome || "");
        return `<div class="outcome-row ${escapeAttr(active ? "active" : "")}">
          <span>${escapeHtml(item.label)}</span>
          <div><i style="width:${escapeAttr(clampPercent(avg))}%"></i></div>
          <b>${escapeHtml(formatNumber(avg))}%</b>
          <small>${escapeHtml(item.team)} · σ ${escapeHtml(formatNumber(summary.std ?? 0))} · rango ${escapeHtml(formatNumber(summary.min ?? 0))}-${escapeHtml(formatNumber(summary.max ?? 0))}</small>
        </div>`;
      }).join("")}
    </div>
  </section>`;
}

function reportConsensusScoreHtml(distribution) {
  const payload = distribution || {};
  if (!payload.available) return "";
  const lambdas = payload.lambdas || {};
  const topScores = payload.top_scores || [];
  return `<section class="report-panel consensus-score-panel">
    <header>
      <strong>Matriz consenso de marcador</strong>
      <small>${escapeHtml(payload.model_count || 0)} modelos · λ ${escapeHtml(formatNumber(lambdas.home ?? "-"))}/${escapeHtml(formatNumber(lambdas.away ?? "-"))}</small>
    </header>
    <div class="top-scores compact">
      ${topScores.slice(0, 5).map((score) => `<span>${escapeHtml(score.score)} <b>${escapeHtml(formatNumber(score.probability ?? 0))}%</b></span>`).join("")}
    </div>
    ${scoreHeatmapHtml(payload)}
  </section>`;
}

function reportTotalsStatsHtml(stats, consensus) {
  const totals = ((stats || {}).totals) || {};
  const consensusTotals = (consensus && consensus.totals) || {};
  const lines = ["0.5", "1.5", "2.5", "3.5"];
  return `<section class="report-panel">
    <header><strong>Over/Under</strong><small>Acuerdo y variabilidad por línea</small></header>
    <div class="totals-list stat-totals">
      ${lines.map((line) => {
        const item = totals[line] || {};
        const fallback = consensusTotals[line] || {};
        const over = item.over || {};
        const under = item.under || {};
        const label = item.label || fallback.label || "-";
        const share = Math.round(Number(item.share ?? fallback.share ?? 0) * 100);
        return `<div class="total-row">
          <span>U/O ${escapeHtml(line)}</span>
          <b>${escapeHtml(label)} · ${escapeHtml(share)}%</b>
          <small>O ${escapeHtml(formatNumber(over.avg ?? 0))}% σ${escapeHtml(formatNumber(over.std ?? 0))} · U ${escapeHtml(formatNumber(under.avg ?? 0))}% σ${escapeHtml(formatNumber(under.std ?? 0))}</small>
        </div>`;
      }).join("")}
    </div>
  </section>`;
}

function allModelsDetailsHtml(models) {
  const items = models || [];
  if (!items.length) return "";
  return `<details class="models-drawer">
    <summary>Todos los modelos (${escapeHtml(items.length)})</summary>
    <div class="model-consensus-list">
      ${items.map((model) => modelConsensusRowHtml(model)).join("")}
    </div>
  </details>`;
}

function reportTotalsConsensusHtml(consensus) {
  const totals = consensus.totals || {};
  const lines = ["0.5", "1.5", "2.5", "3.5"];
  return `<div class="totals-list">
    ${lines.map((line) => {
      const item = totals[line] || {};
      return `<div class="total-row">
        <span>U/O ${escapeHtml(line)}</span>
        <b>${escapeHtml(item.label || "-")}</b>
        <b>${escapeHtml(Math.round(Number(item.share || 0) * 100))}%</b>
      </div>`;
    }).join("")}
  </div>`;
}

function agreementMatrixHtml(models) {
  return `<div class="agreement-matrix">
    ${(models || []).map((model) => {
      const decision = model.decision || {};
      return `<div class="agreement-cell">
        <span>${escapeHtml(model.model_label || model.model_key || "")}</span>
        <strong>${escapeHtml(decision.label || "-")} · ${escapeHtml(model.top_score || "-")}</strong>
      </div>`;
    }).join("")}
  </div>`;
}

function modelConsensusRowHtml(model) {
  const decision = model.decision || {};
  const expected = model.expected_goals || {};
  const warnings = model["warnings"] || [];
  const probs = model.probabilities || {};
  const confidence = probs[decision.outcome] ?? "";
  return `<div class="model-consensus-row ${escapeAttr(model.fallback ? "fallback" : "")}">
    <strong>${escapeHtml(model.model_label || model.model_key || "")}</strong>
    <b>${escapeHtml(decision.label || "-")}${confidence !== "" ? ` · ${escapeHtml(formatNumber(confidence))}%` : ""}</b>
    <span>${escapeHtml(model.top_score || "-")}</span>
    <span>λ ${escapeHtml(expected.home ?? "-")} / ${escapeHtml(expected.away ?? "-")}</span>
    <span>${escapeHtml(model.consensus_eligible ? "Cuenta" : "Excluido")}${warnings.length ? ` · ${escapeHtml(warnings[0])}` : ""}</span>
  </div>`;
}

function renderUpcomingPredictions(result) {
  const summary = result.summary || {};
  const recentLimit = summary.poisson_recent_matches || currentPoissonRecentMatches();
  document.getElementById("upcoming-summary").textContent =
    `${summary.returned || 0}/${summary.requested || 0} partidos - ${summary.group || "Todos"} - Poisson ultimos ${recentLimit} - ML ${summary.use_ml_model ? "activo" : "off"}`;
  document.getElementById("upcoming-predictions").innerHTML = (result.predictions || []).map((prediction) => {
    const fixture = prediction.fixture || {};
    const probs = prediction.probabilities || {};
    const sources = prediction.market_sources || {};
    const contextual = prediction.contextual_poisson || {};
    const homeAsset = assetFor(fixture.home || "");
    const awayAsset = assetFor(fixture.away || "");
    const outcomes = [
      { key: "home", label: "1", team: fixture.home || "Local", value: probs.home ?? 0 },
      { key: "draw", label: "X", team: "Empate", value: probs.draw ?? 0 },
      { key: "away", label: "2", team: fixture.away || "Visitante", value: probs.away ?? 0 },
    ];
    const totals = [
      { label: "0.5", over: probs.over05, under: probs.under05 },
      { label: "1.5", over: probs.over15, under: probs.under15 },
      { label: "2.5", over: probs.over25, under: probs.under25 },
      { label: "3.5", over: probs.over35, under: probs.under35 },
    ];
    const favorite = [...outcomes].sort((a, b) => Number(b.value || 0) - Number(a.value || 0))[0] || outcomes[0];
    return `<article class="upcoming-card">
      <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || "")}</strong></header>
      <div class="upcoming-match">
        <div class="upcoming-team">${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
        <span>vs</span>
        <div class="upcoming-team away">${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
      </div>
      <div class="prediction-pick">
        <span>Favorito</span>
        <strong>${escapeHtml(favorite.label)} · ${escapeHtml(favorite.team)}</strong>
      </div>
      <div class="outcome-list">
        ${outcomes.map((item) => `
          <div class="outcome-row ${escapeAttr(item.key === favorite.key ? "active" : "")}">
            <span>${escapeHtml(item.label)}</span>
            <div><i style="width:${escapeAttr(clampPercent(item.value))}%"></i></div>
            <b>${escapeHtml(item.value)}%</b>
          </div>`).join("")}
      </div>
      <div class="totals-list">
        ${totals.map((line) => `
          <div class="total-row">
            <span>U/O ${escapeHtml(line.label)}</span>
            <b>O ${escapeHtml(line.over ?? "-")}%</b>
            <b>U ${escapeHtml(line.under ?? "-")}%</b>
          </div>`).join("")}
      </div>
      ${contextualPoissonHtml(contextual, fixture)}
      <div class="source-strip">
        <span>${marketBadgeText(sources.result, "1X2: Poisson")}</span>
        ${goalMarketLines.map((line) => `<span>${marketBadgeText(sources[line.key], `${line.label}: Poisson`)}</span>`).join("")}
      </div>
      ${marketReadoutHtml(prediction.market_readout || {})}
      <small>${escapeHtml((prediction.notes || []).join(" - "))}</small>
    </article>`;
  }).join("") || loadingHtml("Sin fixtures futuros");
  renderTable("upcoming-predictions-table", result.table);
}

function marketReadoutHtml(readout) {
  const lines = (readout && readout.lines) || [];
  if (lines.length) {
    const ranked = [...lines].sort((a, b) => Math.abs(Number(b.raw_edge || 0)) - Math.abs(Number(a.raw_edge || 0))).slice(0, 4);
    return `<div class="market-readout">
      ${ranked.map((line) => `
        <span>${escapeHtml(line.market || "")} ${escapeHtml(line.label || "")}: modelo ${escapeHtml(line.model_probability ?? "-")}% · mercado ${escapeHtml(line.market_probability ?? "-")}% · edge ${escapeHtml(line.raw_edge ?? "-")}pp</span>
      `).join("")}
    </div>`;
  }
  const missing = (readout && readout.missing_sources) || [];
  if (!missing.length) return "";
  return `<div class="market-readout muted">${missing.map((item) => `<span>Fuente faltante: ${escapeHtml(item)}</span>`).join("")}</div>`;
}

function contextualPoissonHtml(contextual, fixture) {
  const context = contextual || {};
  const fixtureData = fixture || {};
  const probs = context.probabilities || {};
  const topScores = context.top_scores || [];
  const overUnder = context.over_under || {};
  const recentLimit = Number(context.match_limit || currentPoissonRecentMatches() || 15);
  const recentLabel = `Poisson ultimos ${recentLimit}`;
  const hasMatrix = Boolean(context.available || context.matrix_available || topScores.length || ((context.heatmap || {}).cells || []).length);
  if (!hasMatrix) {
    return `<section class="context-poisson unavailable">
      <header><strong>${escapeHtml(recentLabel)}</strong><small>${escapeHtml(context.reason || "all_matches.csv no disponible")}</small></header>
    </section>`;
  }
  const title = context.available ? recentLabel : "Poisson base";
  const lambdaText = `λ ${context.context_lambda_home ?? "-"} / ${context.context_lambda_away ?? "-"}`;
  const detail = context.available ? lambdaText : `${context.reason || "recent15 no disponible"} · ${lambdaText}`;
  return `<section class="context-poisson">
    <header>
      <strong>${escapeHtml(title)}</strong>
      <small>${escapeHtml(detail)}</small>
    </header>
    <div class="context-outcomes">
      <span>1 <b>${escapeHtml(probs.home ?? "-")}%</b></span>
      <span>X <b>${escapeHtml(probs.draw ?? "-")}%</b></span>
      <span>2 <b>${escapeHtml(probs.away ?? "-")}%</b></span>
    </div>
    <div class="context-totals">
      ${Object.entries(overUnder).map(([line, values]) => `
        <span>${escapeHtml(line)} <b>O ${escapeHtml(values.over ?? "-")}%</b><b>U ${escapeHtml(values.under ?? "-")}%</b></span>
      `).join("")}
    </div>
    <div class="top-scores">
      ${topScores.map((score) => `<span>${escapeHtml(score.score)} <b>${escapeHtml(score.probability)}%</b></span>`).join("")}
    </div>
    ${scoreHeatmapHtml(context)}
    ${context.available ? `<details class="recent15-drawer">
      <summary>Ultimos ${escapeHtml(recentLimit)} partidos</summary>
      <div class="recent15-columns">
        ${recentMatchesMiniTable((context.recent_matches || {}).home || [], fixtureData.home || "Local")}
        ${recentMatchesMiniTable((context.recent_matches || {}).away || [], fixtureData.away || "Visitante")}
      </div>
    </details>` : ""}
  </section>`;
}

function scoreHeatmapHtml(contextual) {
  const heatmap = (contextual && contextual.heatmap) || {};
  const homeGoals = heatmap.home_goals || [];
  const awayGoals = heatmap.away_goals || [];
  const cells = heatmap.cells || [];
  if (!homeGoals.length || !awayGoals.length || !cells.length) return "";
  const cellMap = new Map(cells.map((cell) => [`${cell.home_goals}-${cell.away_goals}`, cell]));
  const maxProb = Math.max(Number(heatmap.max_probability || 0), 0.001);
  const header = `<span></span>${awayGoals.map((goal) => `<b>${escapeHtml(goal)}</b>`).join("")}`;
  const rows = homeGoals.map((homeGoal) => `
    <b>${escapeHtml(homeGoal)}</b>
    ${awayGoals.map((awayGoal) => {
      const cell = cellMap.get(`${homeGoal}-${awayGoal}`) || {};
      const heat = Math.max(0.04, Math.min(1, Number(cell.probability || 0) / maxProb));
      return `<span title="${escapeAttr(cell.score || "")}: ${escapeAttr(cell.probability ?? 0)}%" style="--heat:${escapeAttr(heat)}">${escapeHtml(cell.probability ?? "")}</span>`;
    }).join("")}
  `).join("");
  return `<div class="score-heatmap" style="grid-template-columns: 24px repeat(${awayGoals.length}, minmax(28px, 1fr))">${header}${rows}</div>`;
}

function recentMatchesMiniTable(rows, team) {
  const items = rows || [];
  if (!items.length) return `<div class="recent15-table"><strong>${escapeHtml(team)}</strong><small>Sin partidos recientes</small></div>`;
  return `<div class="recent15-table">
    <strong>${escapeHtml(team)}</strong>
    <table>
      <thead><tr><th>Fecha</th><th>Rival</th><th>Marcador</th><th>Tipo</th></tr></thead>
      <tbody>${items.map((row) => `
        <tr>
          <td>${escapeHtml(row.date || "")}</td>
          <td>${escapeHtml(row.opponent || "")}</td>
          <td>${escapeHtml(row.score || "")}</td>
          <td>${escapeHtml(row.match_type || "")}</td>
        </tr>`).join("")}</tbody>
    </table>
  </div>`;
}

function metricsTableFromModel(model) {
  const metrics = (model && model.metrics) || {};
  const rows = Object.entries(metrics).map(([split, values]) => ({ Split: split, ...(values || {}) }));
  const columns = rows.length ? Object.keys(rows[0]) : [];
  return { columns, rows, total: rows.length };
}

async function runMatchMonteCarlo() {
  clearAlert();
  const button = document.getElementById("simulate-poisson-btn");
  const limit = Number(document.getElementById("sim-match-limit").value || 8);
  const group = document.getElementById("sim-group-filter").value || "";
  const iterations = currentMonteCarloSimulations();
  document.getElementById("simulation-summary").textContent =
    `Simulando ${limit} partido${limit === 1 ? "" : "s"} con ${formatInteger(iterations)} simulaciones Monte Carlo...`;
  if (button) button.disabled = true;
  try {
    const result = await api("/api/mundial/monte-carlo-matches", jsonOptions({
      ...simulationPayload({ iterations, use_ml_model: false, mode: "poisson_live" }),
      limit,
      group,
    }));
    renderMatchMonteCarlo(result);
  } catch (error) {
    document.getElementById("match-simulation-grid").innerHTML = loadingHtml("Monte Carlo no disponible");
    document.getElementById("simulation-summary").textContent = "";
    showError(error.message);
  } finally {
    if (button) button.disabled = false;
  }
}

function renderMatchMonteCarlo(result) {
  const summary = result.summary || {};
  const scoreLabel = summary.score_model_label || summary.score_model || "Poisson independiente";
  document.getElementById("simulation-summary").textContent =
    `${summary.method || "Monte Carlo por partido"} - ${summary.returned || 0}/${summary.requested || 0} partidos - ${formatInteger(summary.iterations || 0)} simulaciones - seed ${summary.seed || ""} - ${scoreLabel} - Poisson ultimos ${summary.poisson_recent_matches || currentPoissonRecentMatches()}`;
  document.getElementById("match-simulation-grid").innerHTML = (result.predictions || [])
    .map((prediction) => monteCarloMatchCardHtml(prediction))
    .join("") || loadingHtml("Sin fixtures para simular");
  renderTable("match-simulation-table", result.table);
}

function monteCarloMatchCardHtml(prediction) {
  const fixture = prediction.fixture || {};
  const probs = prediction.probabilities || {};
  const expected = prediction.expected_goals || {};
  const simulated = prediction.simulated_goals || {};
  const scoreModel = prediction.score_model || {};
  const topScores = prediction.top_scores || [];
  const homeAsset = assetFor(fixture.home || "");
  const awayAsset = assetFor(fixture.away || "");
  const outcomes = [
    { key: "home", label: "1", team: fixture.home || "Local", value: probs.home ?? 0 },
    { key: "draw", label: "X", team: "Empate", value: probs.draw ?? 0 },
    { key: "away", label: "2", team: fixture.away || "Visitante", value: probs.away ?? 0 },
  ];
  const totals = [
    { label: "0.5", over: probs.over05, under: probs.under05 },
    { label: "1.5", over: probs.over15, under: probs.under15 },
    { label: "2.5", over: probs.over25, under: probs.under25 },
    { label: "3.5", over: probs.over35, under: probs.under35 },
  ];
  const favorite = [...outcomes].sort((a, b) => Number(b.value || 0) - Number(a.value || 0))[0] || outcomes[0];
  return `<article class="upcoming-card monte-carlo-card">
    <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || "")}</strong></header>
    <div class="upcoming-match">
      <div class="upcoming-team">${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
      <span>vs</span>
      <div class="upcoming-team away">${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
    </div>
    <div class="prediction-pick">
      <span>Monte Carlo · ${escapeHtml(formatInteger(prediction.iterations || 0))} simulaciones</span>
      <strong>${escapeHtml(favorite.label)} · ${escapeHtml(favorite.team)}</strong>
    </div>
    <div class="outcome-list">
      ${outcomes.map((item) => `
        <div class="outcome-row ${escapeAttr(item.key === favorite.key ? "active" : "")}">
          <span>${escapeHtml(item.label)}</span>
          <div><i style="width:${escapeAttr(clampPercent(item.value))}%"></i></div>
          <b>${escapeHtml(item.value)}%</b>
        </div>`).join("")}
    </div>
    <div class="totals-list">
      ${totals.map((line) => `
        <div class="total-row">
          <span>U/O ${escapeHtml(line.label)}</span>
          <b>O ${escapeHtml(line.over ?? "-")}%</b>
          <b>U ${escapeHtml(line.under ?? "-")}%</b>
        </div>`).join("")}
    </div>
    <div class="top-scores">
      ${topScores.map((score) => `<span>${escapeHtml(score.score)} <b>${escapeHtml(score.probability)}%</b></span>`).join("")}
    </div>
    <div class="source-strip">
      <span>λ ${escapeHtml(expected.home ?? "-")} / ${escapeHtml(expected.away ?? "-")}</span>
      <span>Media sim ${escapeHtml(simulated.home ?? "-")} / ${escapeHtml(simulated.away ?? "-")}</span>
      <span>N=${escapeHtml(formatInteger(prediction.iterations || 0))}</span>
      <span>${escapeHtml(scoreModel.label || prediction.source || "Poisson")}</span>
    </div>
    ${contextualPoissonHtml(prediction.contextual_poisson || {}, fixture)}
  </article>`;
}

async function runSimulation(mode = "hybrid") {
  clearAlert();
  const poissonLive = mode === "poisson_live";
  document.getElementById("simulation-summary").textContent = poissonLive ? "Ejecutando Monte Carlo Poisson live..." : "Ejecutando Monte Carlo...";
  try {
    const simMlToggle = document.getElementById("sim-use-ml-model");
    const job = await api("/api/mundial/simulate", jsonOptions(simulationPayload({
      mode,
      include_confirmed_results: poissonLive,
      use_ml_model: poissonLive ? false : Boolean(simMlToggle && simMlToggle.checked),
    })));
    trackWorldcupJob(job, "simulation");
  } catch (error) {
    document.getElementById("simulation-summary").textContent = "";
    showError(error.message);
  }
}

function trackWorldcupJob(job, kind) {
  if (!job || !job.job_id) return;
  job.kind = kind;
  job.handled = false;
  state.jobs.set(job.job_id, job);
  setWorldcupJobBusy(kind, true);
  renderWorldcupJobProgress(kind);
  startWorldcupJobPolling();
}

function startWorldcupJobPolling() {
  if (state.jobTimer || state.jobPollingInFlight) return;
  scheduleWorldcupJobPoll(0);
}

function scheduleWorldcupJobPoll(delay) {
  if (state.jobTimer) window.clearTimeout(state.jobTimer);
  state.jobTimer = window.setTimeout(pollWorldcupJobs, Math.max(Number(delay) || 0, 0));
}

async function pollWorldcupJobs() {
  if (state.jobPollingInFlight) return;
  if (state.jobTimer) {
    window.clearTimeout(state.jobTimer);
    state.jobTimer = null;
  }
  state.jobPollingInFlight = true;
  let hasActive = false;
  let nextDelay = 10000;
  try {
    for (const jobId of [...state.jobs.keys()]) {
      const previous = state.jobs.get(jobId) || {};
      if (isTerminalJob(previous) && previous.handled) continue;
      try {
        const job = await api(`/api/jobs/${jobId}`);
        job.kind = previous.kind;
        job.handled = previous.handled;
        applyWorldcupJobPollState(job, previous);
        if (isTerminalJob(job) && !job.handled) {
          job.handled = true;
          state.jobs.set(jobId, job);
          await handleWorldcupJobComplete(job);
        } else {
          state.jobs.set(jobId, job);
        }
        if (!isTerminalJob(job)) {
          hasActive = true;
          nextDelay = Math.min(nextDelay, worldcupJobPollDelay(job));
        }
        renderWorldcupJobProgress(job.kind);
      } catch (error) {
        previous.status = "failed";
        previous.error = error.message;
        previous.handled = true;
        state.jobs.set(jobId, previous);
        await handleWorldcupJobComplete(previous);
        renderWorldcupJobProgress(previous.kind);
      }
    }
  } finally {
    state.jobPollingInFlight = false;
  }
  for (const job of state.jobs.values()) {
    if (!isTerminalJob(job)) {
      hasActive = true;
      nextDelay = Math.min(nextDelay, worldcupJobPollDelay(job));
    }
  }
  if (hasActive) {
    scheduleWorldcupJobPoll(nextDelay);
  }
}

function applyWorldcupJobPollState(job, previous) {
  const nextSignature = worldcupJobProgressSignature(job);
  const previousSignature = previous.pollSignature || worldcupJobProgressSignature(previous);
  job.pollSignature = nextSignature;
  job.pollIdleCount = nextSignature === previousSignature ? Number(previous.pollIdleCount || 0) + 1 : 0;
}

function worldcupJobProgressSignature(job) {
  const progress = job.progress || {};
  return [
    job.status || "",
    progress.stage || "",
    progress.message || "",
    progress.current ?? "",
    progress.total ?? "",
    progress.percent ?? "",
    progress.current_trial ?? "",
    progress.total_trials ?? "",
    progress.model_index ?? "",
    progress.model_total ?? "",
    progress.model_key ?? "",
    progress.fixture_index ?? "",
    progress.fixture_total ?? "",
    job.updated_at || "",
  ].join("|");
}

function worldcupJobPollDelay(job) {
  const progress = job.progress || {};
  const kind = job.kind || "";
  const stage = progress.stage || "";
  if (kind === "upcoming-report") return 1000;
  const base = kind === "simulation" ? 2000 : stage === "tuning" ? 5000 : 3000;
  const idleCount = Number(job.pollIdleCount || 0);
  if (idleCount >= 4) return 10000;
  if (idleCount >= 2) return Math.min(base * 2, 10000);
  return base;
}

async function handleWorldcupJobComplete(job) {
  setWorldcupJobBusy(job.kind, false);
  renderWorldcupJobProgress(job.kind);
  if (job.status === "failed") {
    showError(job.error || "Proceso fallido");
    if (job.kind === "training" && state.training) renderTrainingStatus(state.training);
    return;
  }
  const result = job.result || {};
  if (job.kind === "training") {
    state.newModelMode = false;
    state.activeModelId = (result.model || {}).model_id || result.active_model_id || state.activeModelId;
    renderTrainingResult(result);
    if (result.models) renderModelsCatalog(result.models);
    else await loadModelsCatalog();
    await loadTrainingStatus();
    if (state.activeModelId) {
      const active = state.models.find((model) => model.model_id === state.activeModelId) || result.model || {};
      state.training = { ...(state.training || {}), model: active };
      renderTrainingStatus(state.training);
      renderActiveModel(active);
      renderTrainingControls(state.trainingOptions, active);
      const modelIdInput = document.getElementById("worldcup-model-id");
      if (modelIdInput && active.model_id) {
        modelIdInput.value = active.model_id;
        modelIdInput.dataset.autofilled = "true";
      }
      renderModelState(active, state.training || {});
    }
    const simMlToggle = document.getElementById("sim-use-ml-model");
    if (simMlToggle) simMlToggle.checked = true;
    document.getElementById("simulation-summary").textContent = `Modelo listo: ${(result.model || {}).model_name || (result.model || {}).model_id || "híbrido"}`;
  }
  if (job.kind === "simulation") {
    renderSimulation(result);
  }
  if (job.kind === "upcoming-report") {
    renderUpcomingReport(result);
  }
}

function renderWorldcupJobProgress(kind) {
  const targetId = kind === "training"
    ? "worldcup-training-progress"
    : kind === "simulation"
      ? "worldcup-simulation-progress"
      : kind === "upcoming-report"
        ? "worldcup-upcoming-progress"
        : "";
  if (!targetId) return;
  const target = document.getElementById(targetId);
  if (!target) return;
  const job = latestWorldcupJob(kind);
  if (!job) {
    target.className = "worldcup-progress hidden";
    target.innerHTML = "";
    return;
  }
  const progress = job.progress || {};
  const percent = clampPercent(progress.percent ?? (job.status === "succeeded" ? 100 : 0));
  const current = progress.current_trial || progress.current || 0;
  const total = progress.total_trials || progress.total || 0;
  const label = progress.message || job.message || jobLabels[job.status] || job.status;
  const best = progress.best_value === "" || progress.best_value === null || progress.best_value === undefined
    ? ""
    : `<span>Mejor ${escapeHtml(formatNumber(progress.best_value))}</span>`;
  const stateText = progress.last_state ? `<span>${escapeHtml(progress.last_state)}</span>` : "";
  const market = progress.market ? `<span>${escapeHtml(progress.market)}</span>` : "";
  const throughput = progress.rows_per_second ? `<span>${escapeHtml(progress.rows_per_second)} filas/s</span>` : "";
  const eta = progress.eta_seconds ? `<span>ETA ${escapeHtml(formatElapsed(progress.eta_seconds))}</span>` : "";
  const modelStep = progress.model_total ? `<span>Modelo ${escapeHtml(progress.model_index || 0)}/${escapeHtml(progress.model_total)}</span>` : "";
  const fixtureStep = progress.fixture_total ? `<span>Fixture ${escapeHtml(progress.fixture_index || 0)}/${escapeHtml(progress.fixture_total)}</span>` : "";
  const hardware = progress.hardware ? `<span>${escapeHtml((progress.hardware || {}).actual_device || "cpu")}</span>` : "";
  const error = job.error ? `<span>${escapeHtml(cleanMessage(job.error))}</span>` : "";
  const activity = worldcupJobActivityLabel(job);
  target.className = `worldcup-progress ${escapeAttr(job.status || "queued")}`;
  target.innerHTML = `
    <div class="progress-header">
      <div class="progress-title">
        <strong>${escapeHtml(jobLabels[job.status] || job.status || "Proceso")}</strong>
        <small>${escapeHtml(label)}</small>
      </div>
      <strong>${escapeHtml(percent)}%</strong>
    </div>
    <div class="progress-bar"><div class="progress-fill" style="width:${escapeAttr(percent)}%"></div></div>
    <div class="progress-meta">
      <span>${escapeHtml(progress.stage || job.status || "queued")}</span>
      <span>${escapeHtml(current)}/${escapeHtml(total)}</span>
      ${market}
      ${throughput}
      ${eta}
      ${modelStep}
      ${fixtureStep}
      ${hardware}
      ${best}
      ${stateText}
      ${activity ? `<span>${escapeHtml(activity)}</span>` : ""}
      ${error}
    </div>`;
}

function worldcupJobActivityLabel(job) {
  if (!job || !job.updated_at || isTerminalJob(job)) return "";
  const seconds = secondsSinceIso(job.updated_at);
  if (seconds === null) return "";
  if (seconds >= 20) return `Procesando lote; última actualización hace ${formatElapsed(seconds)}`;
  return `Actualizado hace ${formatElapsed(seconds)}`;
}

function secondsSinceIso(value) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return null;
  return Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
}

function formatElapsed(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  const remainder = total % 60;
  if (minutes < 60) return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const minuteRemainder = minutes % 60;
  return minuteRemainder ? `${hours}h ${minuteRemainder}m` : `${hours}h`;
}

function latestWorldcupJob(kind) {
  return [...state.jobs.values()].filter((job) => job.kind === kind).pop();
}

function isTerminalJob(job) {
  return job && ["succeeded", "failed"].includes(job.status);
}

function setWorldcupJobBusy(kind, busy) {
  const ids = kind === "simulation"
    ? ["simulate-poisson-btn"]
    : kind === "upcoming-report"
      ? ["upcoming-predict-btn"]
      : ["training-train", "training-retrain-base"];
  ids.forEach((id) => {
    const button = document.getElementById(id);
    if (button) button.disabled = Boolean(busy);
  });
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.min(100, Math.max(0, Math.round(number)));
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value;
  return number.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatInteger(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value;
  return Math.round(number).toLocaleString("es-MX");
}

function predictionCard(label, value) {
  return `<article class="prediction-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>`;
}

function marketSourceValue(source, fallback) {
  return (source && source.source) || fallback;
}

function marketBadgeText(source, fallback) {
  if (!source || !source.source) return escapeHtml(fallback);
  const model = source.model_name ? ` - ${source.model_name}` : "";
  return `${escapeHtml(source.label || "")}: ${escapeHtml(source.source)}${escapeHtml(model)}`;
}

function trainingPayload(walkForwardMode = "none") {
  const payload = {
    ...simulationPayload(),
    model_id: document.getElementById("worldcup-model-id").value || "",
    model_name: document.getElementById("worldcup-model-id").value || "",
    model_type: document.getElementById("worldcup-model-type").value || "xgboost",
    market_mode: "dual_markets",
    training_target: "result",
    walk_forward_mode: walkForwardMode,
    device: document.getElementById("worldcup-device").value || "auto",
    n_jobs: Number(document.getElementById("worldcup-n-jobs").value || -1),
    tuning_enabled: document.getElementById("worldcup-tuning-enabled").checked,
    n_trials: Number(document.getElementById("worldcup-n-trials").value || 12),
    objective: document.getElementById("worldcup-objective").value || "F1",
    optuna_sampler: document.getElementById("worldcup-optuna-sampler").value || "tpe",
    optuna_pruner: document.getElementById("worldcup-optuna-pruner").value || "none",
    tune_params: document.getElementById("worldcup-tune-params").value || "all",
  };
  const numberFields = {
    n_estimators: "worldcup-n-estimators",
    learning_rate: "worldcup-learning-rate",
    max_depth: "worldcup-max-depth",
    min_child_weight: "worldcup-min-child-weight",
    lambda_regularization: "worldcup-lambda-regularization",
    alpha_regularization: "worldcup-alpha-regularization",
    num_leaves: "worldcup-num-leaves",
    min_child_samples: "worldcup-min-child-samples",
    minibatch_frac: "worldcup-minibatch-frac",
    l2_leaf_reg: "worldcup-l2-leaf-reg",
    random_strength: "worldcup-random-strength",
  };
  Object.entries(numberFields).forEach(([key, id]) => {
    const input = document.getElementById(id);
    if (input && !input.disabled && input.value !== "") payload[key] = Number(input.value);
  });
  const natural = document.getElementById("worldcup-natural-gradient");
  if (natural && !natural.disabled) payload.natural_gradient = natural.checked;
  return payload;
}

function simulationPayload(overrides = {}) {
  const mlWeightInput = document.getElementById("sim-ml-weight");
  const mlToggle = document.getElementById("sim-use-ml-model");
  return {
    model_id: selectedModelId(),
    iterations: currentMonteCarloSimulations(),
    seed: Number(document.getElementById("sim-seed").value || 2026),
    poisson_recent_matches: currentPoissonRecentMatches(),
    history_weight: Number(document.getElementById("sim-history-weight").value || 1),
    recency_weight: Number(document.getElementById("sim-recency-weight").value || 0),
    host_advantage: Number(document.getElementById("sim-host-advantage").value || 45),
    max_goals: Number(document.getElementById("sim-max-goals").value || 10),
    score_model: (document.getElementById("sim-score-model") || {}).value || "independent_poisson",
    ml_weight: Number((mlWeightInput && mlWeightInput.value) || 0.5),
    use_ml_model: Boolean(mlToggle && mlToggle.checked),
    ...overrides,
  };
}

function currentPoissonRecentMatches() {
  const inputs = poissonRecentInputIds.map((id) => document.getElementById(id)).filter(Boolean);
  const input = inputs.find((node) => node.offsetParent !== null) || inputs[0];
  const value = Number(input && input.value ? input.value : 15);
  if (!Number.isFinite(value)) return 15;
  return Math.min(50, Math.max(3, Math.round(value)));
}

function currentMonteCarloSimulations() {
  const input = document.getElementById("sim-iterations");
  const value = Number(input && input.value ? input.value : 5000);
  const simulations = Math.min(100000, Math.max(100, Math.round(Number.isFinite(value) ? value : 5000)));
  if (input && String(input.value) !== String(simulations)) input.value = simulations;
  return simulations;
}

function syncPoissonRecentInputs(source) {
  const value = Math.min(50, Math.max(3, Math.round(Number(source.value || 15) || 15)));
  source.value = value;
  poissonRecentInputIds.forEach((id) => {
    const input = document.getElementById(id);
    if (input && input !== source) input.value = value;
  });
}

function renderSimulation(result) {
  state.lastSimulation = result;
  const summary = result.summary || {};
  const config = summary.config || {};
  const mlState = config.use_ml_model ? "ML híbrido activo" : "ML híbrido off";
  const layers = (summary.hybrid_layers || []).join(" + ");
  const recentLimit = config.poisson_recent_matches || currentPoissonRecentMatches();
  const scoreModel = summary.score_model || {};
  document.getElementById("simulation-summary").textContent =
    `${summary.model || "Modelo"} - ${config.iterations || ""} iteraciones - seed ${config.seed || ""} - ${scoreModel.label || config.score_model || "Poisson independiente"} - Poisson ultimos ${recentLimit} - historial ${config.history_weight || ""} - recencia ${config.recency_weight || ""} - ${mlState} - ${layers}`;
  if (!document.getElementById("champion-strip")) return;
  const rows = (result.advancement && result.advancement.rows) || [];
  const topChampions = [...rows].sort((a, b) => Number(b["Campeon %"] || 0) - Number(a["Campeon %"] || 0)).slice(0, 8);
  document.getElementById("champion-strip").innerHTML = topChampions.map((row) => {
    const asset = assetFor(row.Equipo);
    return `<article class="champion-card">
      <div class="team-line">${flagHtml(asset, "large")}<strong>${escapeHtml(row.Equipo)}</strong></div>
      <span>${escapeHtml(row.Grupo)}</span>
      <strong>${escapeHtml(row["Campeon %"])}%</strong>
    </article>`;
  }).join("");
  renderTable("advancement-table", result.advancement);
  renderTable("match-probs-table", result.matches);
  renderQuickSimulationPanel(result);
}

function renderQuickSimulationPanel(result) {
  const panel = document.getElementById("quick-simulation-panel");
  if (!panel) return;
  panel.classList.add("hidden");
  panel.innerHTML = "";
}

function renderProcedure(payload) {
  document.getElementById("procedure-list").innerHTML = (payload.steps || []).map((step, index) => `
    <article class="procedure-step">
      <span>${escapeHtml(index + 1)}</span>
      <div><strong>${escapeHtml(step.name)}</strong><p>${escapeHtml(step.detail)}</p></div>
    </article>`).join("");
}

function rebuildTeamAssets() {
  state.teamAssets.clear();
  state.groups.forEach((group) => {
    (group.teams || []).forEach((team) => state.teamAssets.set(team.name, team));
  });
  state.teams.forEach((row) => {
    if (row.asset) state.teamAssets.set(row.asset.name, row.asset);
  });
}

function assetFor(team) {
  return state.teamAssets.get(team) || { name: team || "", flag_url: "", flag_fallback: initials(team || ""), slug: "" };
}

function flagHtml(asset, size = "") {
  const flag = asset || {};
  const url = flag.flag_url || "";
  const fallbackClass = url ? "visual-fallback" : "visual-fallback visible";
  return `<span class="flag-wrap ${escapeAttr(size)}">
    <span class="${fallbackClass}">${escapeHtml(flag.flag_fallback || initials(flag.name || ""))}</span>
    ${url ? `<img src="${escapeAttr(url)}" alt="Bandera ${escapeAttr(flag.name || "")}" onerror="handleImageError(this)">` : ""}
  </span>`;
}

function playerPhotoHtml(player) {
  const url = player.photo_url || "";
  const fallbackClass = url ? "visual-fallback" : "visual-fallback visible";
  return `<span class="player-photo">
    <span class="${fallbackClass}">${escapeHtml(player.initials || initials(player.name || ""))}</span>
    ${url ? `<img src="${escapeAttr(url)}" alt="${escapeAttr(player.name || "Jugador")}" onerror="handleImageError(this)">` : ""}
  </span>`;
}

function handleImageError(image) {
  const fallback = image.previousElementSibling;
  image.remove();
  if (fallback) fallback.classList.add("visible");
}
window.handleImageError = handleImageError;

function renderTable(id, table) {
  document.getElementById(id).innerHTML = tableHtml(table);
}

function tableHtml(table) {
  if (!table || !table.columns) return "<div></div>";
  const head = table.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const rows = (table.rows || []).map((row) => `<tr>${table.columns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div><small>${escapeHtml(table.total || 0)} filas</small>`;
}

function jsonOptions(payload) {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
}

function switchWorldcupView(id) {
  document.querySelectorAll(".nav-pill").forEach((button) => button.classList.toggle("active", button.dataset.section === id));
  document.querySelectorAll(".worldcup-view").forEach((view) => view.classList.toggle("active", view.id === id));
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (id === "alineaciones" && state.fixtures.length) loadSelectedLineup(false);
  if (id === "predicciones" && state.fixtures.length) fillUpcomingGroupFilter();
}

function loadingHtml(text) {
  return `<div class="fixture-card"><strong>${escapeHtml(text)}</strong></div>`;
}

function shortName(name) {
  const text = String(name || "").trim();
  if (text.length <= 18) return text;
  const parts = text.split(/\s+/);
  if (parts.length >= 2) return `${parts[0][0]}. ${parts[parts.length - 1]}`;
  return text.slice(0, 18);
}

function initials(value) {
  const words = String(value || "").match(/[A-Za-z0-9]+/g) || [];
  if (!words.length) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return words.slice(0, 2).map((word) => word[0]).join("").toUpperCase();
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
    .replace(/^(CLIError|ValueError|RuntimeError|WorldCupTrainingError|LineupProviderError):\s*/, "")
    .replace(/\bNone\b/g, "Sin valor");
}
