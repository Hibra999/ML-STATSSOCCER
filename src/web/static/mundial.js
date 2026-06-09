const state = {
  overview: null,
  groups: [],
  fixtures: [],
  teams: [],
  players: [],
  lineups: [],
  playerFeatures: [],
  teamAssets: new Map(),
  defaultsApplied: false,
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadAll(false);
});

function bindEvents() {
  document.querySelectorAll("[data-section]").forEach((button) => {
    button.addEventListener("click", () => scrollToSection(button.dataset.section));
  });
  document.getElementById("refresh-btn").addEventListener("click", () => loadAll(true));
  document.getElementById("simulate-btn").addEventListener("click", runSimulation);
  document.getElementById("fixture-group-filter").addEventListener("change", renderFixtures);
  document.getElementById("fixture-search").addEventListener("input", renderFixtures);
  document.getElementById("lineup-load").addEventListener("click", () => loadSelectedLineup(false));
  document.getElementById("lineup-autodetect").addEventListener("click", autodetectSelectedLineup);
  document.getElementById("lineup-auto-refresh").addEventListener("click", autoRefreshLineups);
  document.getElementById("lineup-refresh").addEventListener("click", refreshSelectedLineup);
  document.getElementById("lineup-link").addEventListener("click", linkSelectedLineup);
  document.getElementById("lineup-fixture").addEventListener("change", () => loadSelectedLineup(false));
  document.getElementById("players-refresh").addEventListener("click", () => loadPlayers(true));
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
    const [overview, groups, fixtures, teams, lineups, players, playerFeatures, procedure] = await Promise.all([
      api(`/api/mundial/overview?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/groups?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/fixtures?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/teams?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/lineups?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/players?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/player-features?refresh=${refresh ? "true" : "false"}`),
      api("/api/mundial/procedure"),
    ]);
    state.overview = overview;
    state.groups = groups.groups || [];
    state.fixtures = fixtures.fixtures || [];
    state.teams = teams.teams || [];
    state.lineups = lineups.lineups || [];
    state.players = players.players || [];
    state.playerFeatures = playerFeatures.rows || [];
    rebuildTeamAssets();
    applyDefaultConfig(overview.default_config || {});
    renderOverview(overview);
    renderGroups(groups);
    renderTeams(teams);
    renderFixtureFilters();
    renderFixtures();
    renderLineupsSummary(lineups);
    renderPlayers(players);
    renderPlayerFeatures(playerFeatures);
    renderProcedure(procedure);
    fillLineupSelect();
    await loadSelectedLineup(false);
    await runSimulation();
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
  };
  Object.entries(pairs).forEach(([id, value]) => {
    const input = document.getElementById(id);
    if (input && value !== undefined) input.value = value;
  });
  document.getElementById("sim-use-lineups").checked = Boolean(config.use_lineups);
  document.getElementById("sim-use-player-features").checked = Boolean(config.use_player_features);
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

function simulationPayload() {
  return {
    iterations: Number(document.getElementById("sim-iterations").value || 5000),
    seed: Number(document.getElementById("sim-seed").value || 2026),
    history_weight: Number(document.getElementById("sim-history-weight").value || 1),
    recency_weight: Number(document.getElementById("sim-recency-weight").value || 0),
    host_advantage: Number(document.getElementById("sim-host-advantage").value || 45),
    max_goals: Number(document.getElementById("sim-max-goals").value || 10),
    lineup_weight: Number(document.getElementById("sim-lineup-weight").value || 1),
    player_feature_weight: Number(document.getElementById("sim-player-feature-weight").value || 1),
    use_lineups: document.getElementById("sim-use-lineups").checked,
    use_player_features: document.getElementById("sim-use-player-features").checked,
  };
}

function renderSimulation(result) {
  const summary = result.summary || {};
  const config = summary.config || {};
  const lineupState = config.use_lineups ? "11 activo" : "11 off";
  const featureState = config.use_player_features ? "features XI activas" : "features XI off";
  document.getElementById("simulation-summary").textContent =
    `${summary.model || "Modelo"} - ${config.iterations || ""} iteraciones - seed ${config.seed || ""} - historial ${config.history_weight || ""} - recencia ${config.recency_weight || ""} - ${lineupState} - ${featureState}`;
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

function scrollToSection(id) {
  document.querySelectorAll(".nav-pill").forEach((button) => button.classList.toggle("active", button.dataset.section === id));
  document.getElementById(id).scrollIntoView({ behavior: "smooth", block: "start" });
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
