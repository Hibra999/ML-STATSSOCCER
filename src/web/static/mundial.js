const state = {
  overview: null,
  groups: [],
  fixtures: [],
  teams: [],
  players: [],
  lineups: [],
  playerFeatures: [],
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
  newModelMode: false,
};

const jobLabels = {
  queued: "En cola",
  running: "En proceso",
  succeeded: "Completado",
  failed: "Error",
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadAll(false);
});

function bindEvents() {
  document.querySelectorAll("[data-section]").forEach((button) => {
    button.addEventListener("click", () => switchWorldcupView(button.dataset.section));
  });
  document.getElementById("refresh-btn").addEventListener("click", () => loadAll(true));
  document.getElementById("simulate-btn").addEventListener("click", runSimulation);
  document.getElementById("worldcup-new-model").addEventListener("click", startNewWorldcupModel);
  document.getElementById("model-load").addEventListener("click", loadSelectedModel);
  document.getElementById("model-delete").addEventListener("click", deleteSelectedModel);
  document.getElementById("worldcup-clear-cache").addEventListener("click", clearWorldcupMaintenance);
  document.getElementById("model-active-select").addEventListener("change", syncModelSelects);
  document.getElementById("upcoming-model-select").addEventListener("change", syncModelSelects);
  document.getElementById("fixture-group-filter").addEventListener("change", renderFixtures);
  document.getElementById("fixture-search").addEventListener("input", renderFixtures);
  document.getElementById("lineup-load").addEventListener("click", () => loadSelectedLineup(false));
  document.getElementById("lineup-autodetect").addEventListener("click", autodetectSelectedLineup);
  document.getElementById("lineup-auto-refresh").addEventListener("click", autoRefreshLineups);
  document.getElementById("lineup-refresh").addEventListener("click", refreshSelectedLineup);
  document.getElementById("lineup-link").addEventListener("click", linkSelectedLineup);
  document.getElementById("lineup-fixture").addEventListener("change", () => loadSelectedLineup(false));
  document.getElementById("players-refresh").addEventListener("click", () => loadPlayers(true));
  document.getElementById("training-refresh").addEventListener("click", loadTrainingStatus);
  document.getElementById("training-download").addEventListener("click", downloadTrainingDataset);
  document.getElementById("training-prepare-etl").addEventListener("click", prepareTrainingEtl);
  document.getElementById("training-train").addEventListener("click", trainWorldCupModel);
  document.getElementById("training-retrain-base").addEventListener("click", () => trainWorldCupModel("result_only"));
  document.getElementById("training-retrain-players").addEventListener("click", () => trainWorldCupModel("result_plus_players"));
  document.getElementById("upcoming-predict-btn").addEventListener("click", runUpcomingPredictions);
  document.getElementById("worldcup-model-type").addEventListener("change", () => applyModelDefaults(document.getElementById("worldcup-model-type").value, true));
  document.getElementById("worldcup-model-id").addEventListener("input", (event) => { event.target.dataset.autofilled = "false"; });
  document.getElementById("worldcup-tuning-enabled").addEventListener("change", applyTuningLocks);
  document.getElementById("worldcup-tune-params").addEventListener("input", applyTuningLocks);
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
    const [overview, groups, fixtures, teams, lineups, players, playerFeatures, training, models, procedure] = await Promise.all([
      api(`/api/mundial/overview?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/groups?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/fixtures?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/teams?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/lineups?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/players?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/player-features?refresh=${refresh ? "true" : "false"}`),
      api("/api/mundial/training/status"),
      api("/api/mundial/models"),
      api("/api/mundial/procedure"),
    ]);
    state.overview = overview;
    state.groups = groups.groups || [];
    state.fixtures = fixtures.fixtures || [];
    state.teams = teams.teams || [];
    state.lineups = lineups.lineups || [];
    state.players = players.players || [];
    state.playerFeatures = playerFeatures.rows || [];
    state.training = training;
    state.models = models.models || [];
    state.activeModelId = models.active_model_id || "";
    state.trainingOptions = training.options || null;
    rebuildTeamAssets();
    applyDefaultConfig(overview.default_config || {});
    renderOverview(overview);
    renderGroups(groups);
    renderTeams(teams);
    renderFixtureFilters();
    renderFixtures();
    fillUpcomingGroupFilter();
    renderLineupsSummary(lineups);
    renderPlayers(players);
    renderPlayerFeatures(playerFeatures);
    renderTrainingStatus(training);
    renderModelsCatalog(models);
    renderProcedure(procedure);
    fillLineupSelect();
  } catch (error) {
    showError(error.message);
  }
}

function setLoading() {
  document.getElementById("groups-grid").innerHTML = loadingHtml("Cargando grupos");
  document.getElementById("teams-grid").innerHTML = loadingHtml("Cargando equipos");
  document.getElementById("fixtures-list").innerHTML = loadingHtml("Cargando fixtures");
  document.getElementById("lineup-stage").innerHTML = loadingHtml("Cargando alineaciones");
  document.getElementById("players-list").innerHTML = loadingHtml("Cargando jugadores");
  document.getElementById("lineup-features-table").innerHTML = loadingHtml("Features pendientes");
  document.getElementById("player-features-table").innerHTML = loadingHtml("Features pendientes");
  document.getElementById("training-summary").innerHTML = loadingHtml("Dataset pendiente");
  document.getElementById("training-model-state").innerHTML = loadingHtml("Modelo pendiente");
  document.getElementById("training-etl-flow").innerHTML = loadingHtml("ETL pendiente");
  document.getElementById("training-metric-cards").innerHTML = loadingHtml("Metricas pendientes");
  document.getElementById("training-confusion-matrix").innerHTML = loadingHtml("Matriz pendiente");
  document.getElementById("training-tuning-flow").innerHTML = loadingHtml("Tuning pendiente");
  document.getElementById("training-features").innerHTML = loadingHtml("Features pendientes");
  document.getElementById("training-model-params").innerHTML = loadingHtml("Parametros pendientes");
  document.getElementById("upcoming-predictions").innerHTML = loadingHtml("Predicciones pendientes");
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
    "sim-lineup-weight": config.lineup_weight,
    "sim-player-feature-weight": config.player_feature_weight,
    "sim-ml-weight": config.ml_weight,
  };
  Object.entries(pairs).forEach(([id, value]) => {
    const input = document.getElementById(id);
    if (input && value !== undefined) input.value = value;
  });
  document.getElementById("sim-use-lineups").checked = Boolean(config.use_lineups);
  document.getElementById("sim-use-player-features").checked = Boolean(config.use_player_features);
  document.getElementById("sim-use-ml-model").checked = Boolean(config.use_ml_model);
  state.defaultsApplied = true;
}

function renderOverview(overview) {
  document.getElementById("metric-teams").textContent = overview.teams || 0;
  document.getElementById("metric-groups").textContent = overview.groups || 0;
  document.getElementById("metric-fixtures").textContent = overview.fixtures || 0;
  document.getElementById("metric-players").textContent = overview.players || 0;
  document.getElementById("model-source").textContent = `${overview.model || "Modelo"} - ${overview.fixture_source || ""}`;
  const highlight = overview.highlight || overview.opener || {};
  const kickoffLabel = `${highlight.date || "2026-06-11"} ${highlight.time || ""}`.trim();
  document.getElementById("hero-meta").textContent = highlight.group || highlight.round || "Partido inaugural";
  document.getElementById("hero-match").innerHTML = `
    ${matchTeamHtml(highlight.home || {}, "home")}
    <div class="hero-vs-block">
      <span class="versus">VS</span>
      <div id="hero-countdown" class="hero-countdown hero-countdown-vs"></div>
      <div class="hero-kickoff">
        <strong>${escapeHtml(kickoffLabel || "Horario pendiente")}</strong>
        <small>${escapeHtml(highlight.venue || "Sede por confirmar")}</small>
      </div>
    </div>
    ${matchTeamHtml(highlight.away || {}, "away")}`;
  renderHeroCountdown(overview.countdown_target, overview.countdown_state, highlight);
  renderHeroHardware((state.trainingOptions || {}).hardware || {});
  document.getElementById("hero-next-grid").innerHTML = (overview.next_matches || []).map((fixture) => heroNextCardHtml(fixture)).join("")
    || `<article class="hero-next-card empty"><strong>Sin más partidos cargados</strong><small>El calendario adicional aparecerá aquí.</small></article>`;
}

function matchTeamHtml(asset, side) {
  return `<div class="match-team ${escapeAttr(side)}">
    ${side === "away" ? `<strong>${escapeHtml(asset.name || "")}</strong>` : ""}
    ${flagHtml(asset, "large")}
    ${side !== "away" ? `<strong>${escapeHtml(asset.name || "")}</strong>` : ""}
  </div>`;
}

function renderGroups(payload) {
  document.getElementById("groups-source").textContent = payload.source || "";
  document.getElementById("groups-grid").innerHTML = state.groups.map((group) => `
    <article class="group-card">
      <header><h3>${escapeHtml(group.name)}</h3><strong>${escapeHtml(group.letter)}</strong></header>
      <ol>${(group.teams || []).map((team) => `<li class="team-line">${flagHtml(team)}<strong>${escapeHtml(team.name)}</strong><small>${escapeHtml(team.seed)}</small></li>`).join("")}</ol>
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

function fillLineupSelect() {
  const groupFixtures = state.fixtures.filter((fixture) => fixture.group);
  document.getElementById("lineup-fixture").innerHTML = groupFixtures.map((fixture) => `
    <option value="${escapeAttr(fixture.id)}">${escapeHtml(fixture.id)} - ${escapeHtml(fixture.group)} - ${escapeHtml(fixture.label)}</option>
  `).join("");
}

function fillUpcomingGroupFilter() {
  const groups = [...new Set(state.fixtures.map((fixture) => fixture.group).filter(Boolean))];
  document.getElementById("upcoming-group-filter").innerHTML = `<option value="">Todos los grupos</option>${groups.map((group) => `<option value="${escapeAttr(group)}">${escapeHtml(group)}</option>`).join("")}`;
}

function renderHeroCountdown(targetIso, stateLabel, highlight) {
  const container = document.getElementById("hero-countdown");
  if (state.countdownTimer) {
    window.clearInterval(state.countdownTimer);
    state.countdownTimer = null;
  }
  if (!targetIso) {
    container.innerHTML = `<div class="countdown-chip"><span>Kickoff</span><strong>Hora pendiente</strong></div>`;
    return;
  }
  const render = () => {
    const diff = Date.parse(targetIso) - Date.now();
    if (Number.isNaN(diff)) {
      container.innerHTML = `<div class="countdown-chip"><span>Kickoff</span><strong>Hora pendiente</strong></div>`;
      return;
    }
    if (diff <= 0) {
      container.innerHTML = `<div class="countdown-chip live"><span>${escapeHtml(highlight.group || highlight.round || "Partido")}</span><strong>En curso o ya inició</strong></div>`;
      return;
    }
    const remaining = countdownParts(diff);
    container.innerHTML = [
      countdownChip("Días", remaining.days),
      countdownChip("Horas", remaining.hours),
      countdownChip("Min", remaining.minutes),
      countdownChip("Seg", remaining.seconds),
    ].join("");
  };
  render();
  state.countdownTimer = window.setInterval(render, 1000);
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
  const options = models.map((model) => `<option value="${escapeAttr(model.model_id)}">${escapeHtml(model.model_name || model.model_id)}${model.bundle ? " - 1X2 + O/U" : ""}${model.active ? " - activo" : ""}</option>`).join("");
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
    predictionCard("Eval", evalStrategyLabel(model && model.eval_strategy)),
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
  modelId.value = "";
  modelId.placeholder = `${autoWorldcupModelId(modelType)}-nuevo`;
  modelId.dataset.autofilled = "false";
  document.getElementById("worldcup-tuning-enabled").checked = false;
  applyTuningLocks();
  document.getElementById("sim-use-ml-model").checked = false;
  document.getElementById("upcoming-summary").textContent = "Nuevo modelo pendiente de entrenamiento";
  document.getElementById("upcoming-predictions").innerHTML = loadingHtml("Sin modelo seleccionado");
  document.getElementById("upcoming-predictions-table").innerHTML = "";
  document.getElementById("training-model-state").innerHTML = [
    predictionCard("Modo", "Nuevo modelo"),
    predictionCard("Modelo", "Sin guardar"),
    predictionCard("Mercados", "1X2 + O/U 2.5"),
    predictionCard("Eval", "pendiente"),
  ].join("");
  document.getElementById("training-metric-cards").innerHTML = loadingHtml("Entrena el nuevo modelo");
  document.getElementById("training-confusion-matrix").innerHTML = loadingHtml("Matriz pendiente");
  document.getElementById("training-tuning-flow").innerHTML = tuningFlowHtml({ enabled: false });
  document.getElementById("training-features").innerHTML = loadingHtml("Features pendientes");
  document.getElementById("training-model-params").innerHTML = loadingHtml("Parametros pendientes");
  document.getElementById("simulation-summary").textContent = "Nuevo modelo preparado. Ingresa nombre y parámetros antes de entrenar.";
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
    renderModelsCatalog(result);
    document.getElementById("sim-use-ml-model").checked = true;
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
    document.getElementById("lineup-status").innerHTML = "";
    document.getElementById("lineup-stage").innerHTML = loadingHtml("11 iniciales limpiados");
    document.getElementById("lineups-summary").innerHTML = loadingHtml("Cache de alineaciones limpiado");
    document.getElementById("simulation-summary").textContent = `Limpieza completa: ${((result.removed || []).length)} rutas procesadas.`;
  } catch (error) {
    showError(error.message);
  }
}

async function loadSelectedLineup(refresh) {
  const fixtureId = document.getElementById("lineup-fixture").value;
  if (!fixtureId) return;
  document.getElementById("lineup-status").innerHTML = `<span>Cargando 11...</span>`;
  try {
    const result = await api(`/api/mundial/fixtures/${encodeURIComponent(fixtureId)}/lineups?refresh=${refresh ? "true" : "false"}`);
    renderLineup(result);
    await loadFixturePlayerStats(fixtureId, false);
  } catch (error) {
    document.getElementById("lineup-status").innerHTML = "";
    showError(error.message);
  }
}

async function autodetectSelectedLineup() {
  const fixtureId = document.getElementById("lineup-fixture").value;
  if (!fixtureId) return;
  document.getElementById("lineup-status").innerHTML = `<span>Buscando evento SofaScore...</span>`;
  try {
    const result = await api(`/api/mundial/fixtures/${encodeURIComponent(fixtureId)}/autodetect`, jsonOptions({ fetch_lineup: true }));
    const event = result.event || {};
    if (event.match_url) document.getElementById("lineup-url").value = event.match_url;
    if (result.lineup && result.lineup.lineup) renderLineup(result.lineup);
    await loadFixturePlayerStats(fixtureId, true);
    await reloadLineupsSummary();
    await loadPlayerFeatures(false);
    document.getElementById("lineup-status").insertAdjacentHTML("beforeend", `<span>Auto match ${escapeHtml(event.confidence || 0)}</span>`);
  } catch (error) {
    showError(error.message);
  }
}

async function autoRefreshLineups() {
  clearAlert();
  document.getElementById("lineup-status").innerHTML = `<span>Detectando eventos y 11 iniciales...</span>`;
  try {
    const result = await api("/api/mundial/lineups/auto-refresh", jsonOptions({ refresh_events: true }));
    await reloadLineupsSummary();
    await loadPlayerFeatures(false);
    await loadSelectedLineup(false);
    document.getElementById("lineup-status").innerHTML = `
      <span>Calendario revisado: ${escapeHtml(result.attempted || 0)}</span>
      <span>Con 11 completo: ${escapeHtml(result.refreshed || 0)}</span>
      <span>Sin detectar: ${escapeHtml(result.failures || 0)}</span>`;
  } catch (error) {
    showError(error.message);
  }
}

async function refreshSelectedLineup() {
  const fixtureId = document.getElementById("lineup-fixture").value;
  if (!fixtureId) return;
  const matchUrl = document.getElementById("lineup-url").value;
  try {
    const result = await api(`/api/mundial/fixtures/${encodeURIComponent(fixtureId)}/lineups/refresh`, jsonOptions({ match_url: matchUrl }));
    renderLineup(result);
    await loadFixturePlayerStats(fixtureId, true);
    await reloadLineupsSummary();
    await loadPlayerFeatures(false);
  } catch (error) {
    showError(error.message);
  }
}

async function linkSelectedLineup() {
  const fixtureId = document.getElementById("lineup-fixture").value;
  const matchUrl = document.getElementById("lineup-url").value.trim();
  if (!fixtureId || !matchUrl) {
    showError("Pega la URL SofaScore del partido");
    return;
  }
  try {
    const result = await api(`/api/mundial/fixtures/${encodeURIComponent(fixtureId)}/lineups/link`, jsonOptions({ match_url: matchUrl, refresh: true }));
    renderLineup(result);
    await loadFixturePlayerStats(fixtureId, true);
    await reloadLineupsSummary();
    await loadPlayerFeatures(false);
  } catch (error) {
    showError(error.message);
  }
}

async function loadFixturePlayerStats(fixtureId, refresh) {
  try {
    const result = await api(`/api/mundial/fixtures/${encodeURIComponent(fixtureId)}/player-stats?refresh=${refresh ? "true" : "false"}`);
    renderTable("lineup-features-table", result.features);
    return result;
  } catch (error) {
    document.getElementById("lineup-features-table").innerHTML = loadingHtml("Features no disponibles");
    return null;
  }
}

async function loadPlayerFeatures(refresh) {
  const result = await api(`/api/mundial/player-features?refresh=${refresh ? "true" : "false"}`);
  state.playerFeatures = result.rows || [];
  renderPlayerFeatures(result);
  return result;
}

function renderPlayerFeatures(payload) {
  renderTable("player-features-table", payload.features);
}

async function reloadLineupsSummary() {
  const payload = await api("/api/mundial/lineups");
  state.lineups = payload.lineups || [];
  renderLineupsSummary(payload);
}

function renderLineup(result) {
  const lineup = result.lineup || {};
  document.getElementById("lineup-url").value = lineup.match_url || document.getElementById("lineup-url").value || "";
  document.getElementById("lineup-source").textContent = lineup.source || "";
  document.getElementById("lineup-status").innerHTML = `
    <span>${escapeHtml(lineup.status || "Pendiente")}</span>
    <span>${escapeHtml(lineup.home || "")}: ${escapeHtml(lineup.starters_home || 0)}/11</span>
    <span>${escapeHtml(lineup.away || "")}: ${escapeHtml(lineup.starters_away || 0)}/11</span>
    ${lineup.error ? `<span>${escapeHtml(lineup.error)}</span>` : ""}`;
  document.getElementById("lineup-stage").innerHTML = `
    ${lineupSideHtml(lineup, "home", lineup.home, lineup.formation_home, lineup.home_asset)}
    ${lineupSideHtml(lineup, "away", lineup.away, lineup.formation_away, lineup.away_asset)}`;
  renderTable("lineup-table", result.players);
}

function lineupSideHtml(lineup, side, team, formation, asset) {
  const starters = (lineup.players || []).filter((player) => player.team === team && player.starter).slice(0, 11);
  return `<article class="lineup-side">
    <header>
      <div class="lineup-title">${flagHtml(asset || assetFor(team))}<div><strong>${escapeHtml(team || "Equipo")}</strong><small>${escapeHtml(side === "home" ? "Local" : "Visitante")}</small></div></div>
      <span class="formation-badge">${escapeHtml(formation || "Sin 11")}</span>
    </header>
    <div class="pitch">
      ${starters.map((player) => playerTokenHtml(player)).join("") || `<div class="player-token" style="left:50%;top:50%"><strong>11 pendiente</strong><small>Vincula SofaScore</small></div>`}
    </div>
  </article>`;
}

function playerTokenHtml(player) {
  const x = Number(player.x || 50);
  const y = Number(player.y || 50);
  const detail = [player.shirt_number, player.position, player.rating ? `R ${player.rating}` : ""].filter(Boolean).join(" - ");
  const stats = player.stats || {};
  const statDetail = [
    stats.minutesPlayed ? `${stats.minutesPlayed} min` : "",
    stats.goals ? `${stats.goals} gol` : "",
    stats.goalAssist ? `${stats.goalAssist} ast` : "",
  ].filter(Boolean).join(" - ");
  return `<div class="player-token" style="left:${escapeAttr(x)}%;top:${escapeAttr(y)}%">
    ${playerPhotoHtml(player)}
    <strong>${escapeHtml(shortName(player.name || ""))}</strong>
    <small>${escapeHtml(detail)}</small>
    ${statDetail ? `<small>${escapeHtml(statDetail)}</small>` : ""}
  </div>`;
}

function renderLineupsSummary(payload) {
  const rows = payload.lineups || state.lineups;
  document.getElementById("lineups-summary").innerHTML = rows.slice(0, 24).map((row) => `
    <article class="fixture-card">
      <div class="fixture-meta"><span>${escapeHtml(row.date)}</span><span>${escapeHtml(row.status)}</span></div>
      <div class="fixture-teams">
        <div class="fixture-team">${flagHtml(row.home)}<strong>${escapeHtml(row.home.name)}</strong></div>
        <span>${escapeHtml(row.starters_home)}/11</span>
        <div class="fixture-team">${flagHtml(row.away)}<strong>${escapeHtml(row.away.name)}</strong></div>
        <span>${escapeHtml(row.starters_away)}/11</span>
      </div>
    </article>`).join("");
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
  document.getElementById("training-status").textContent = "Descargando Kaggle...";
  try {
    const result = await api("/api/mundial/training/download-kaggle", jsonOptions({ force: false }));
    state.training = result;
    if (!state.trainingOptions) await loadTrainingStatus();
    renderTrainingStatus(result);
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

async function trainWorldCupModel(walkForwardMode = "none") {
  clearAlert();
  if (!state.training || !state.training.etl_ready || state.training.etl_stale) {
    showError("Primero ejecuta Preparar ETL para dejar listo el dataset de entrenamiento.");
    return;
  }
  if (!document.getElementById("worldcup-model-id").value.trim()) {
    showError("Ingresa un nombre para el nuevo modelo antes de entrenar.");
    return;
  }
  const modeLabel = walkForwardMode === "result_plus_players"
    ? "Reentrenando con partido + jugadores..."
    : walkForwardMode === "result_only"
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
  document.getElementById("training-status").textContent = payload.available
    ? `${payload.train_rows || 0} train listo - ${payload.etl_ready ? "ETL listo" : "ETL pendiente"} - ${evalStrategyLabel(payload.eval_strategy)}`
    : "Dataset Kaggle no descargado";
  document.getElementById("training-source").textContent = `${payload.dataset_slug || "Kaggle"} - ${payload.training_mode || "sin modo"} - ${payload.prepared_label_source || "fuente pendiente"}`;
  document.getElementById("training-summary").innerHTML = datasetSummaryHtml(payload);
  const trainDisabled = !payload.etl_ready || payload.etl_stale;
  document.getElementById("training-train").disabled = trainDisabled;
  document.getElementById("training-retrain-base").disabled = trainDisabled;
  renderWalkForwardNotice(payload.walk_forward_refresh || {});
  renderModelState(model, payload);
  renderTable("training-preview", payload.preview);
  renderTrainingWarnings([...(payload.prepared_warnings || []), ...(model.warnings || [])]);
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
    target_column: payload.effective_target,
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
  renderFeatureList(markets);
}

function trainingMarketSections(model, payload) {
  const markets = (model && model.markets) || (payload && payload.markets) || {};
  const keys = ["result", "over_under_25"].filter((key) => markets[key]);
  if (keys.length) {
    return keys.map((key) => ({ key, label: markets[key].label || marketLabel(key), ...markets[key] }));
  }
  const target = (model && (model.effective_target || model.requested_target)) || (payload && (payload.effective_target || payload.requested_target)) || "result";
  return [{
    key: target === "over_under_25" ? "over_under_25" : "result",
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
  const evalValue = payload.test_rows
    ? `${payload.test_rows} filas test`
    : `${payload.eval_rows || 0} holdout`;
  const walkForward = payload.walk_forward || {};
  const refresh = payload.walk_forward_refresh || {};
  return [
    datasetCard("Archivos", (payload.files || []).length, "CSV/XLS detectados"),
    datasetCard("ETL", payload.etl_ready ? (payload.etl_stale ? "Desactualizado" : "Listo") : "Pendiente", payload.prepared_label_source || "preparar artifact"),
    datasetCard("Train etiquetado", payload.train_rows || 0, payload.training_mode || "sin modo"),
    datasetCard("Evaluacion", evalValue, evalStrategyLabel(payload.eval_strategy)),
    datasetCard("Predicción 2026", payload.prediction_rows || 0, "filas sin label usadas como features"),
    datasetCard("Features equipo", payload.team_feature_rows || 0, "equipos disponibles"),
    datasetCard("Walk-forward", walkForward.matches || 0, `${refresh.ready_result_only || 0} base / ${refresh.ready_with_players || 0} con jugadores`),
    datasetCard("O/U 2.5", payload.prepared_over_under_ready ? "Listo" : "Pendiente", "solo con goles reales"),
    datasetCard("Target", payload.target_column || "-", "label entrenable"),
  ].join("");
}

function datasetCard(label, value, detail) {
  return `<article class="dataset-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail || "")}</small></article>`;
}

function renderModelState(model, payload) {
  document.getElementById("training-model-state").innerHTML = [
    predictionCard("Modelo", model.trained ? (model.model_label || payload.model_type || "Listo") : "Pendiente"),
    predictionCard("Mercados", modelMarketLabel(model.trained ? model : payload)),
    predictionCard("Eval", evalStrategyLabel(model.eval_strategy || payload.eval_strategy)),
    predictionCard("Walk-forward", walkForwardModeLabel((model.walk_forward_mode || (model.walk_forward_summary || {}).mode || "none"))),
  ].join("");
}

function modelMarketLabel(model) {
  if (!model) return "-";
  if (model.bundle || model.market_mode === "dual_markets" || model.requested_target === "dual_markets") {
    return model.market_models && !model.market_models.over_under_25 ? "1X2 + O/U Poisson" : "1X2 + O/U 2.5";
  }
  const target = model.effective_target || model.requested_target || model.training_target || "";
  if (target === "over_under_25") return "O/U 2.5";
  if (target === "team_strength") return "1X2 team-strength";
  if (target === "result") return "1X2";
  return target || "-";
}

function walkForwardModeLabel(mode) {
  if (mode === "result_plus_players") return "Partido + jugadores";
  if (mode === "result_only") return "Partido base";
  return "Sin incremental";
}

function marketLabel(key) {
  if (key === "over_under_25") return "O/U 2.5";
  if (key === "team_strength") return "1X2 team-strength";
  return "1X2";
}

function evalStrategyLabel(strategy) {
  if (strategy === "final_worldcup_test") return "ultimo Mundial test";
  if (strategy === "test_file") return "test etiquetado";
  if (strategy === "holdout_temporal") return "holdout temporal";
  if (strategy === "holdout_from_train") return "holdout desde train";
  if (strategy === "unavailable") return "sin evaluacion";
  return strategy || "pendiente";
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
  document.getElementById("worldcup-device").value = (model.hardware || {}).requested_device || (options.defaults || {}).device || "auto";
  document.getElementById("worldcup-n-jobs").value = (model.hardware || {}).n_jobs ?? (options.defaults || {}).n_jobs ?? -1;
  if (!state.trainingControlsApplied) {
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
    modelIdInput.value = autoWorldcupModelId(modelKey);
    modelIdInput.dataset.autofilled = "true";
  }
  applyTuningLocks();
}

function autoWorldcupModelId(modelKey) {
  const shortModel = { xgboost: "xgb", lightgbm: "lgbm", catboost: "cat", ngboost: "ngb" }[modelKey] || modelKey || "model";
  return `mundial-${shortModel}-hibrido`;
}

function renderTrainingWarnings(warnings) {
  document.getElementById("training-warnings").innerHTML = (warnings || []).map((warning) => `<span>${escapeHtml(warning)}</span>`).join("");
}

function renderWalkForwardNotice(refresh) {
  const container = document.getElementById("training-walkforward-notice");
  if (!container) return;
  const items = [];
  if (refresh.requires_reload) items.push(`Recarga pendiente: ${refresh.stale_match_ids?.length || 0} partidos jugados sin snapshot.`);
  if (refresh.ready_result_only) items.push(`Reentreno base listo: ${refresh.ready_result_only}`);
  if (refresh.ready_with_players) items.push(`Reentreno + jugadores listo: ${refresh.ready_with_players}`);
  if (refresh.latest_played_fixture) items.push(`Último jugado: ${refresh.latest_played_fixture}`);
  if (refresh.note && !items.includes(refresh.note)) items.push(refresh.note);
  container.innerHTML = items.map((item) => `<span>${escapeHtml(item)}</span>`).join("") || `<span>Sin alertas de walk-forward.</span>`;
  const playersButton = document.getElementById("training-retrain-players");
  if (playersButton) playersButton.disabled = !(refresh.ready_with_players > 0);
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
      <section class="market-panel">
        <header><strong>${escapeHtml(market.label || "Mercado")}</strong><small>${escapeHtml(confusionTargetLabel(market.effective_target || market.key || ""))}</small></header>
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
  const header = `<div></div>${labels.map((label) => `<strong>${escapeHtml(label)}</strong>`).join("")}`;
  const rows = matrix.map((row, rowIndex) => `
    <strong>${escapeHtml(labels[rowIndex])}</strong>
    ${row.map((value, colIndex) => {
      const intensity = Math.max(0.12, Number(value || 0) / maxValue);
      const correct = rowIndex === colIndex ? " correct" : "";
      return `<span class="confusion-cell${correct}" style="--intensity:${escapeAttr(intensity)}"><b>${escapeHtml(value)}</b></span>`;
    }).join("")}
  `).join("");
  return `<div class="confusion-grid" style="grid-template-columns: 120px repeat(${labels.length}, minmax(82px, 1fr))">${header}${rows}</div>`;
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

function renderFeatureList(markets) {
  if (Array.isArray(markets)) {
    document.getElementById("training-features").innerHTML = markets.map((market) => `
      <section class="market-panel">
        <header><strong>${escapeHtml(market.label || "Mercado")}</strong><small>${escapeHtml(market.model_id || "")}</small></header>
        ${featureListHtml(market.top_features || [])}
      </section>`).join("");
    return;
  }
  document.getElementById("training-features").innerHTML = featureListHtml(markets || []);
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
  if (target === "over_under_25") return "2 clases: Under / Over";
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
  const limit = Number(document.getElementById("upcoming-predict-limit").value || 8);
  const group = document.getElementById("upcoming-group-filter").value || "";
  const modelId = document.getElementById("upcoming-model-select").value || selectedModelId();
  document.getElementById("upcoming-summary").textContent = "Calculando próximos partidos...";
  try {
    const result = await api("/api/mundial/predict-upcoming", jsonOptions({ ...simulationPayload(), model_id: modelId, limit, group }));
    renderUpcomingPredictions(result);
  } catch (error) {
    document.getElementById("upcoming-predictions").innerHTML = loadingHtml("Predicciones no disponibles");
    showError(error.message);
  }
}

function renderUpcomingPredictions(result) {
  const summary = result.summary || {};
  document.getElementById("upcoming-summary").textContent = `${summary.returned || 0}/${summary.requested || 0} partidos - ${summary.group || "Todos"} - ML ${summary.use_ml_model ? "activo" : "off"}`;
  document.getElementById("upcoming-predictions").innerHTML = (result.predictions || []).map((prediction) => {
    const fixture = prediction.fixture || {};
    const probs = prediction.probabilities || {};
    const sources = prediction.market_sources || {};
    const homeAsset = assetFor(fixture.home || "");
    const awayAsset = assetFor(fixture.away || "");
    return `<article class="upcoming-card">
      <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || "")}</strong></header>
      <div class="upcoming-match">
        <div class="upcoming-team">${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
        <span>vs</span>
        <div class="upcoming-team away">${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
      </div>
      <div class="prob-strip">
        <span>1 <b>${escapeHtml(probs.home ?? "")}%</b></span>
        <span>X <b>${escapeHtml(probs.draw ?? "")}%</b></span>
        <span>2 <b>${escapeHtml(probs.away ?? "")}%</b></span>
      </div>
      <div class="prob-strip muted">
        <span>O2.5 <b>${escapeHtml(probs.over25 ?? "")}%</b></span>
        <span>U2.5 <b>${escapeHtml(probs.under25 ?? "")}%</b></span>
        <span>Score <b>${escapeHtml(prediction.modal_score || "")}</b></span>
      </div>
      <div class="source-strip">
        <span>${marketBadgeText(sources.result, "1X2: Poisson")}</span>
        <span>${marketBadgeText(sources.over_under_25, "O/U: Poisson")}</span>
      </div>
      <small>${escapeHtml((prediction.notes || []).join(" - "))}</small>
    </article>`;
  }).join("") || loadingHtml("Sin fixtures futuros");
  renderTable("upcoming-predictions-table", result.table);
}

function metricsTableFromModel(model) {
  const metrics = (model && model.metrics) || {};
  const rows = Object.entries(metrics).map(([split, values]) => ({ Split: split, ...(values || {}) }));
  const columns = rows.length ? Object.keys(rows[0]) : [];
  return { columns, rows, total: rows.length };
}

async function runSimulation() {
  clearAlert();
  document.getElementById("simulation-summary").textContent = "Ejecutando Monte Carlo...";
  try {
    const job = await api("/api/mundial/simulate", jsonOptions(simulationPayload()));
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
  if (kind === "simulation") renderWorldcupJobProgress(kind);
  startWorldcupJobPolling();
}

function startWorldcupJobPolling() {
  if (state.jobTimer) return;
  state.jobTimer = window.setInterval(pollWorldcupJobs, 1000);
  pollWorldcupJobs();
}

async function pollWorldcupJobs() {
  let hasActive = false;
  for (const jobId of [...state.jobs.keys()]) {
    const previous = state.jobs.get(jobId) || {};
    if (isTerminalJob(previous) && previous.handled) continue;
    try {
      const job = await api(`/api/jobs/${jobId}`);
      job.kind = previous.kind;
      job.handled = previous.handled;
      if (isTerminalJob(job) && !job.handled) {
        job.handled = true;
        state.jobs.set(jobId, job);
        await handleWorldcupJobComplete(job);
      } else {
        state.jobs.set(jobId, job);
      }
      if (!isTerminalJob(job)) hasActive = true;
      if (job.kind === "simulation") renderWorldcupJobProgress(job.kind);
    } catch (error) {
      previous.status = "failed";
      previous.error = error.message;
      previous.handled = true;
      state.jobs.set(jobId, previous);
      await handleWorldcupJobComplete(previous);
      if (previous.kind === "simulation") renderWorldcupJobProgress(previous.kind);
    }
  }
  if (!hasActive && state.jobTimer) {
    window.clearInterval(state.jobTimer);
    state.jobTimer = null;
  }
}

async function handleWorldcupJobComplete(job) {
  setWorldcupJobBusy(job.kind, false);
  if (job.status === "failed") {
    showError(job.error || "Proceso fallido");
    if (job.kind === "training" && state.training) renderTrainingStatus(state.training);
    return;
  }
  const result = job.result || {};
  if (job.kind === "training") {
    state.newModelMode = false;
    renderTrainingResult(result);
    await loadTrainingStatus();
    if (result.models) renderModelsCatalog(result.models);
    else await loadModelsCatalog();
    document.getElementById("sim-use-ml-model").checked = true;
    document.getElementById("simulation-summary").textContent = `Modelo listo: ${(result.model || {}).model_name || (result.model || {}).model_id || "híbrido"}`;
  }
  if (job.kind === "simulation") {
    renderSimulation(result);
  }
}

function renderWorldcupJobProgress(kind) {
  if (kind !== "simulation") return;
  const target = document.getElementById("worldcup-simulation-progress");
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
  const error = job.error ? `<span>${escapeHtml(cleanMessage(job.error))}</span>` : "";
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
      ${best}
      ${stateText}
      ${error}
    </div>`;
}

function latestWorldcupJob(kind) {
  return [...state.jobs.values()].filter((job) => job.kind === kind).pop();
}

function isTerminalJob(job) {
  return job && ["succeeded", "failed"].includes(job.status);
}

function setWorldcupJobBusy(kind, busy) {
  const ids = kind === "simulation"
    ? ["simulate-btn"]
    : ["training-train", "training-retrain-base", "training-retrain-players"];
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

function simulationPayload() {
  return {
    model_id: selectedModelId(),
    iterations: Number(document.getElementById("sim-iterations").value || 5000),
    seed: Number(document.getElementById("sim-seed").value || 2026),
    history_weight: Number(document.getElementById("sim-history-weight").value || 1),
    recency_weight: Number(document.getElementById("sim-recency-weight").value || 0),
    host_advantage: Number(document.getElementById("sim-host-advantage").value || 45),
    max_goals: Number(document.getElementById("sim-max-goals").value || 10),
    lineup_weight: Number(document.getElementById("sim-lineup-weight").value || 1),
    player_feature_weight: Number(document.getElementById("sim-player-feature-weight").value || 1),
    ml_weight: Number(document.getElementById("sim-ml-weight").value || 0.5),
    use_lineups: document.getElementById("sim-use-lineups").checked,
    use_player_features: document.getElementById("sim-use-player-features").checked,
    use_ml_model: document.getElementById("sim-use-ml-model").checked,
  };
}

function renderSimulation(result) {
  const summary = result.summary || {};
  const config = summary.config || {};
  const lineupState = config.use_lineups ? "11 activo" : "11 off";
  const featureState = config.use_player_features ? "features XI activas" : "features XI off";
  const mlState = config.use_ml_model ? "ML híbrido activo" : "ML híbrido off";
  const layers = (summary.hybrid_layers || []).join(" + ");
  document.getElementById("simulation-summary").textContent =
    `${summary.model || "Modelo"} - ${config.iterations || ""} iteraciones - seed ${config.seed || ""} - historial ${config.history_weight || ""} - recencia ${config.recency_weight || ""} - ${lineupState} - ${featureState} - ${mlState} - ${layers}`;
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
