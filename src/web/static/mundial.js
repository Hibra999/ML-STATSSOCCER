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
  document.getElementById("model-load").addEventListener("click", loadSelectedModel);
  document.getElementById("model-delete").addEventListener("click", deleteSelectedModel);
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
  document.getElementById("training-train").addEventListener("click", trainWorldCupModel);
  document.getElementById("predict-match-btn").addEventListener("click", runMatchPrediction);
  document.getElementById("upcoming-predict-btn").addEventListener("click", runUpcomingPredictions);
  document.getElementById("worldcup-model-type").addEventListener("change", () => applyModelDefaults(document.getElementById("worldcup-model-type").value, true));
  document.getElementById("worldcup-target").addEventListener("change", () => applyModelDefaults(document.getElementById("worldcup-model-type").value, true));
  document.getElementById("worldcup-model-id").addEventListener("input", (event) => { event.target.dataset.autofilled = "false"; });
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
    fillPredictSelect();
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
  document.getElementById("training-hardware").innerHTML = loadingHtml("Hardware pendiente");
  document.getElementById("training-etl-flow").innerHTML = loadingHtml("ETL pendiente");
  document.getElementById("training-metric-cards").innerHTML = loadingHtml("Metricas pendientes");
  document.getElementById("training-confusion-matrix").innerHTML = loadingHtml("Matriz pendiente");
  document.getElementById("training-tuning-flow").innerHTML = loadingHtml("Tuning pendiente");
  document.getElementById("training-features").innerHTML = loadingHtml("Features pendientes");
  document.getElementById("training-model-params").innerHTML = loadingHtml("Parametros pendientes");
  document.getElementById("match-prediction").innerHTML = loadingHtml("Prediccion pendiente");
  document.getElementById("match-prob-breakdown").innerHTML = loadingHtml("Desglose pendiente");
  document.getElementById("upcoming-predictions").innerHTML = loadingHtml("Predicciones pendientes");
  document.getElementById("active-model-state").innerHTML = loadingHtml("Modelo pendiente");
  document.getElementById("models-list").innerHTML = loadingHtml("Modelos pendientes");
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
  const opener = overview.opener || {};
  document.getElementById("hero-meta").textContent = `${opener.date || "2026-06-11"} ${opener.time || ""} - ${opener.venue || "Sede por confirmar"}`;
  document.getElementById("hero-match").innerHTML = `
    ${matchTeamHtml(opener.home || {}, "home")}
    <span class="versus">VS</span>
    ${matchTeamHtml(opener.away || {}, "away")}`;
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
        <span>vs</span>
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

function fillPredictSelect() {
  const groupFixtures = state.fixtures.filter((fixture) => fixture.group);
  document.getElementById("predict-fixture").innerHTML = groupFixtures.map((fixture) => `
    <option value="${escapeAttr(fixture.id)}">${escapeHtml(fixture.id)} - ${escapeHtml(fixture.date)} - ${escapeHtml(fixture.label)}</option>
  `).join("");
}

function fillUpcomingGroupFilter() {
  const groups = [...new Set(state.fixtures.map((fixture) => fixture.group).filter(Boolean))];
  document.getElementById("upcoming-group-filter").innerHTML = `<option value="">Todos los grupos</option>${groups.map((group) => `<option value="${escapeAttr(group)}">${escapeHtml(group)}</option>`).join("")}`;
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
  const options = models.map((model) => `<option value="${escapeAttr(model.model_id)}">${escapeHtml(model.model_name || model.model_id)}${model.active ? " - activo" : ""}</option>`).join("");
  const selectHtml = options || `<option value="">Sin modelos entrenados</option>`;
  ["model-active-select", "upcoming-model-select"].forEach((id) => {
    const select = document.getElementById(id);
    if (!select) return;
    select.innerHTML = selectHtml;
    select.value = activeId || (models[0] || {}).model_id || "";
  });
  const active = models.find((model) => model.model_id === activeId) || models[0] || {};
  renderActiveModel(active);
  document.getElementById("models-list").innerHTML = models.map((model) => `
    <article class="model-row ${model.active ? "active" : ""}">
      <div><strong>${escapeHtml(model.model_name || model.model_id)}</strong><small>${escapeHtml(model.model_id)} - ${escapeHtml(model.model_label || model.model_type || "")}</small></div>
      <span>${escapeHtml(model.effective_target || "-")}</span>
      <b>${model.active ? "Activo" : ""}</b>
    </article>`).join("") || loadingHtml("Entrena tu primer modelo Mundial");
}

function renderActiveModel(model) {
  document.getElementById("active-model-state").innerHTML = [
    predictionCard("Activo", model && model.trained ? (model.model_name || model.model_id) : "Sin modelo"),
    predictionCard("Tipo", (model && (model.model_label || model.model_type)) || "-"),
    predictionCard("Target", (model && model.effective_target) || "-"),
    predictionCard("Eval", evalStrategyLabel(model && model.eval_strategy)),
  ].join("");
}

function syncModelSelects(event) {
  const value = event && event.target ? event.target.value : selectedModelId();
  ["model-active-select", "upcoming-model-select"].forEach((id) => {
    const select = document.getElementById(id);
    if (select && value) select.value = value;
  });
}

function selectedModelId() {
  const activeSelect = document.getElementById("model-active-select");
  const upcomingSelect = document.getElementById("upcoming-model-select");
  return (activeSelect && activeSelect.value) || (upcomingSelect && upcomingSelect.value) || state.activeModelId || "";
}

async function loadSelectedModel() {
  clearAlert();
  const modelId = selectedModelId();
  if (!modelId) {
    showError("Entrena o selecciona un modelo Mundial primero.");
    return;
  }
  try {
    const result = await api("/api/mundial/models/select", jsonOptions({ model_id: modelId }));
    renderModelsCatalog(result);
    document.getElementById("sim-use-ml-model").checked = true;
    document.getElementById("simulation-summary").textContent = `Modelo activo: ${result.selected.model_name || result.selected.model_id}`;
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
      <span>Fixtures revisados: ${escapeHtml(result.attempted || 0)}</span>
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

async function trainWorldCupModel() {
  clearAlert();
  document.getElementById("training-status").textContent = "Entrenando modelo Mundial...";
  try {
    const result = await api("/api/mundial/models/train", jsonOptions(trainingPayload()));
    renderTrainingResult(result);
    await loadTrainingStatus();
    if (result.models) renderModelsCatalog(result.models);
    else await loadModelsCatalog();
    document.getElementById("sim-use-ml-model").checked = true;
    await runMatchPrediction();
  } catch (error) {
    showError(error.message);
  }
}

function renderTrainingStatus(payload) {
  const model = payload.model || {};
  state.trainingOptions = payload.options || state.trainingOptions;
  renderTrainingControls(state.trainingOptions, model);
  renderHardware((model.hardware && model.trained) ? model.hardware : ((state.trainingOptions || {}).hardware || {}));
  document.getElementById("training-status").textContent = payload.available
    ? `${payload.train_rows || 0} train etiquetado - ${evalStrategyLabel(payload.eval_strategy)} - ${payload.prediction_rows || 0} prediccion`
    : "Dataset Kaggle no descargado";
  document.getElementById("training-source").textContent = `${payload.dataset_slug || "Kaggle"} - ${payload.training_mode || "sin modo"}`;
  document.getElementById("training-summary").innerHTML = datasetSummaryHtml(payload);
  renderModelState(model, payload);
  renderTable("training-preview", payload.preview);
  renderTable("training-metrics", metricsTableFromModel(model));
  renderTrainingWarnings(model.warnings || []);
  renderTrainingVisuals(model, payload);
  renderTable("training-model-params", paramsTable(model));
}

function renderTrainingResult(payload) {
  renderTable("training-metrics", payload.metrics_table);
  renderHardware(payload.hardware || {});
  renderTrainingWarnings(payload.warnings || []);
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
  renderTable("training-model-params", paramsTable(payload.model || {}));
}

function renderTrainingVisuals(model, payload) {
  renderEtlFlow((model.etl_steps || payload.etl_steps || []));
  renderMetricCards(model.metrics || payload.metrics || {});
  renderConfusionMatrix(model.confusion_matrix || payload.confusion_matrix || {});
  renderTuningFlow(model.tuning_trace || payload.tuning_trace || model.tuning || {});
  renderFeatureList(model.top_features || []);
}

function datasetSummaryHtml(payload) {
  const evalValue = payload.test_rows
    ? `${payload.test_rows} filas test`
    : `${payload.eval_rows || 0} holdout`;
  return [
    datasetCard("Archivos", (payload.files || []).length, "CSV/XLS detectados"),
    datasetCard("Train etiquetado", payload.train_rows || 0, payload.training_mode || "sin modo"),
    datasetCard("Evaluacion", evalValue, evalStrategyLabel(payload.eval_strategy)),
    datasetCard("Prediccion 2026", payload.prediction_rows || 0, "filas sin label usadas como features"),
    datasetCard("Features equipo", payload.team_feature_rows || 0, "equipos disponibles"),
    datasetCard("Target", payload.target_column || "-", "label entrenable"),
  ].join("");
}

function datasetCard(label, value, detail) {
  return `<article class="dataset-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail || "")}</small></article>`;
}

function renderModelState(model, payload) {
  document.getElementById("training-model-state").innerHTML = [
    predictionCard("Modelo", model.trained ? (model.model_label || payload.model_type || "Listo") : "Pendiente"),
    predictionCard("Target efectivo", model.effective_target || payload.effective_target || "-"),
    predictionCard("Eval", evalStrategyLabel(model.eval_strategy || payload.eval_strategy)),
    predictionCard("Clases", ((model.classes || []).join("/") || "-")),
  ].join("");
}

function evalStrategyLabel(strategy) {
  if (strategy === "test_file") return "test etiquetado";
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
  const target = model.requested_target || (options.defaults || {}).training_target || "result";
  document.getElementById("worldcup-target").value = target;
  const modelId = model.model_id || autoWorldcupModelId(selectedModel, target);
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
      return;
    }
    input.disabled = false;
    if (force || input.value === "") input.value = defaults[key];
  });
  const natural = document.getElementById("worldcup-natural-gradient");
  natural.disabled = defaults.natural_gradient === undefined;
  natural.checked = Boolean(defaults.natural_gradient);
  const target = document.getElementById("worldcup-target").value || "result";
  const modelIdInput = document.getElementById("worldcup-model-id");
  if (modelIdInput && (force || !modelIdInput.value || modelIdInput.dataset.autofilled !== "false")) {
    modelIdInput.value = autoWorldcupModelId(modelKey, target);
    modelIdInput.dataset.autofilled = "true";
  }
}

function autoWorldcupModelId(modelKey, target) {
  const shortModel = { xgboost: "xgb", lightgbm: "lgbm", catboost: "cat", ngboost: "ngb" }[modelKey] || modelKey || "model";
  const shortTarget = target === "over_under_25" ? "uo25" : "result";
  return `mundial-${shortModel}-${shortTarget}`;
}

function renderHardware(hardware) {
  const devices = hardware.cuda_devices || [];
  document.getElementById("training-hardware").innerHTML = [
    predictionCard("CPU cores", hardware.cpu_count || "-"),
    predictionCard("CUDA", hardware.cuda_available ? "Disponible" : "No disponible"),
    predictionCard("Device real", hardware.actual_device || hardware.device_default || "cpu"),
    predictionCard("n_jobs", hardware.n_jobs ?? hardware.default_n_jobs ?? "-1"),
    predictionCard("Threads", hardware.effective_n_jobs || hardware.cpu_count || "-"),
  ].join("");
  if (!hardware.cuda_available && hardware.cuda_error) {
    document.getElementById("training-hardware").insertAdjacentHTML("beforeend", `<small class="hardware-note">${escapeHtml(hardware.cuda_error)}</small>`);
  }
  if (devices.length) {
    document.getElementById("training-hardware").insertAdjacentHTML("beforeend", `<small class="hardware-note">${devices.map(escapeHtml).join(" | ")}</small>`);
  }
}

function renderTrainingWarnings(warnings) {
  document.getElementById("training-warnings").innerHTML = (warnings || []).map((warning) => `<span>${escapeHtml(warning)}</span>`).join("");
}

function renderEtlFlow(steps) {
  document.getElementById("training-etl-flow").innerHTML = (steps || []).map((step, index) => `
    <article class="etl-step ${escapeAttr(step.status || "info")}">
      <span>${escapeHtml(index + 1)}</span>
      <div><strong>${escapeHtml(step.name || "")}</strong><small>${escapeHtml(step.detail || "")}</small></div>
      <b>${escapeHtml(step.count ?? "")}</b>
    </article>`).join("") || loadingHtml("ETL pendiente");
}

function renderMetricCards(metrics) {
  const evalMetrics = (metrics && (metrics.eval || metrics.Eval)) || {};
  const rows = ["Accuracy", "F1", "Precision", "Recall"].map((key) => predictionCard(key, evalMetrics[key] ?? "-"));
  document.getElementById("training-metric-cards").innerHTML = rows.join("");
}

function renderConfusionMatrix(payload) {
  const labels = payload.labels || [];
  const matrix = payload.matrix || [];
  if (!labels.length || !matrix.length) {
    document.getElementById("training-confusion-matrix").innerHTML = loadingHtml("Matriz pendiente");
    return;
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
  document.getElementById("training-confusion-matrix").innerHTML = `<div class="confusion-grid" style="grid-template-columns: 120px repeat(${labels.length}, minmax(82px, 1fr))">${header}${rows}</div>`;
}

function renderTuningFlow(trace) {
  const steps = trace.steps || [];
  const head = trace.enabled
    ? `<div class="tuning-head"><strong>Best ${escapeHtml(trace.objective || "")}: ${escapeHtml(trace.best_value ?? "")}</strong><small>Trial ${escapeHtml(trace.best_trial ?? "")} - ${escapeHtml(trace.trials ?? "")} trials</small></div>`
    : `<div class="tuning-head"><strong>Fine-tuning desactivado</strong><small>Se usaron parametros manuales/default.</small></div>`;
  const items = steps.map((step) => `<article class="tuning-step ${escapeAttr(step.status || "info")}"><strong>${escapeHtml(step.name || "")}</strong><small>${escapeHtml(step.detail || "")}</small></article>`).join("");
  document.getElementById("training-tuning-flow").innerHTML = head + `<div class="tuning-steps">${items}</div>`;
}

function renderFeatureList(features) {
  document.getElementById("training-features").innerHTML = (features || []).slice(0, 10).map((item) => `
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
    return `<article class="upcoming-card">
      <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || "")}</strong></header>
      <div class="upcoming-match">
        <strong>${escapeHtml(fixture.home || "")}</strong>
        <span>vs</span>
        <strong>${escapeHtml(fixture.away || "")}</strong>
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
    const result = await api("/api/mundial/simulate", jsonOptions(simulationPayload()));
    renderSimulation(result);
  } catch (error) {
    document.getElementById("simulation-summary").textContent = "";
    showError(error.message);
  }
}

async function runMatchPrediction() {
  const fixtureId = document.getElementById("predict-fixture").value || document.getElementById("lineup-fixture").value;
  try {
    const result = await api("/api/mundial/predict-match", jsonOptions({ ...simulationPayload(), fixture_id: fixtureId }));
    renderMatchPrediction(result);
  } catch (error) {
    document.getElementById("match-prediction").innerHTML = loadingHtml("Prediccion no disponible");
    showError(error.message);
  }
}

function renderMatchPrediction(result) {
  const fixture = result.fixture || {};
  const probs = result.probabilities || {};
  document.getElementById("match-prediction").innerHTML = [
    predictionCard(`1 - ${fixture.home || "Local"}`, `${probs.home || 0}%`),
    predictionCard("X - Empate", `${probs.draw || 0}%`),
    predictionCard(`2 - ${fixture.away || "Visitante"}`, `${probs.away || 0}%`),
    predictionCard("Over 2.5", `${probs.over25 || 0}%`),
    predictionCard("Under 2.5", `${probs.under25 || 0}%`),
  ].join("");
  renderTable("match-prediction-detail", objectTable({
    Partido: `${fixture.home || ""} vs ${fixture.away || ""}`,
    Fecha: fixture.date || "",
    Prediccion: result.prediction || "",
    Marcador: result.modal_score || "",
    "xG local": (result.expected_goals || {}).home || "",
    "xG visita": (result.expected_goals || {}).away || "",
    Nota: (result.notes || []).join(" - "),
  }));
  renderTable("match-prob-breakdown", predictionBreakdownTable(result));
}

function predictionBreakdownTable(result) {
  const model = result.model_probs || {};
  const poisson = model.poisson || {};
  const poissonTotals = model.poisson_totals || {};
  const ml = model.ml || {};
  const overMl = model.over_under_ml || {};
  const final = result.probabilities || {};
  const rows = [
    {
      Fuente: "Poisson 1X2",
      "1": poisson.H ?? "",
      "X": poisson.D ?? "",
      "2": poisson.A ?? "",
      Over: poissonTotals.over25 ?? "",
      Under: poissonTotals.under25 ?? "",
      Peso: "base",
    },
    {
      Fuente: "ML 1X2",
      "1": ml.H ?? "",
      "X": ml.D ?? "",
      "2": ml.A ?? "",
      Over: "",
      Under: "",
      Peso: model.result_weight ?? 0,
    },
    {
      Fuente: "ML U/O 2.5",
      "1": "",
      "X": "",
      "2": "",
      Over: overMl.over25 ?? "",
      Under: overMl.under25 ?? "",
      Peso: model.over_under_weight ?? 0,
    },
    {
      Fuente: "Final blend",
      "1": final.home ?? "",
      "X": final.draw ?? "",
      "2": final.away ?? "",
      Over: final.over25 ?? "",
      Under: final.under25 ?? "",
      Peso: model.ml_weight ?? 0,
    },
  ];
  return { columns: ["Fuente", "1", "X", "2", "Over", "Under", "Peso"], rows, total: rows.length };
}

function predictionCard(label, value) {
  return `<article class="prediction-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>`;
}

function trainingPayload() {
  const payload = {
    ...simulationPayload(),
    model_id: document.getElementById("worldcup-model-id").value || "",
    model_name: document.getElementById("worldcup-model-id").value || "",
    model_type: document.getElementById("worldcup-model-type").value || "xgboost",
    training_target: document.getElementById("worldcup-target").value || "result",
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
  const mlState = config.use_ml_model ? "Kaggle ML activo" : "Kaggle ML off";
  document.getElementById("simulation-summary").textContent =
    `${summary.model || "Modelo"} - ${config.iterations || ""} iteraciones - seed ${config.seed || ""} - historial ${config.history_weight || ""} - recencia ${config.recency_weight || ""} - ${lineupState} - ${featureState} - ${mlState}`;
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

function objectTable(row) {
  return { columns: Object.keys(row), rows: [row], total: 1 };
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
  if (id === "modelo" && state.models.length) syncModelSelects();
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
    .replace(/^(CLIError|ValueError|RuntimeError|LineupProviderError):\s*/, "")
    .replace(/\bNone\b/g, "Sin valor");
}
