const state = {
  overview: null,
  groups: [],
  fixtures: [],
  teams: [],
  players: [],
  teamAssets: new Map(),
  defaultsApplied: false,
  countdownTimer: null,
  jobs: new Map(),
  jobTimer: null,
  jobPollingInFlight: false,
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

const poissonRecentInputIds = ["sim-poisson-recent-matches", "upcoming-poisson-recent-matches"];

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  loadAll(false);
});

function bindEvents() {
  document.querySelectorAll("[data-section]").forEach((button) => {
    button.addEventListener("click", () => switchWorldcupView(button.dataset.section));
  });
  bind("simulate-poisson-btn", "click", runMatchMonteCarlo);
  bind("fixture-group-filter", "change", renderFixtures);
  bind("fixture-search", "input", renderFixtures);
  bind("players-refresh", "click", () => loadPlayers(true));
  bind("upcoming-predict-btn", "click", runUpcomingPredictions);
  bind("upcoming-pipeline-mode", "change", syncUpcomingPipelineControls);
  poissonRecentInputIds.forEach((id) => {
    const input = document.getElementById(id);
    if (input) input.addEventListener("change", () => syncPoissonRecentInputs(input));
  });
}

function bind(id, event, handler) {
  const node = document.getElementById(id);
  if (node) node.addEventListener(event, handler);
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
    const [overview, groups, fixtures, teams, players, procedure] = await Promise.all([
      api(`/api/mundial/overview?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/groups?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/fixtures?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/teams?refresh=${refresh ? "true" : "false"}`),
      api(`/api/mundial/players?refresh=${refresh ? "true" : "false"}`),
      api("/api/mundial/procedure"),
    ]);
    state.overview = overview;
    state.groups = groups.groups || [];
    state.fixtures = fixtures.fixtures || [];
    state.teams = teams.teams || [];
    state.players = players.players || [];
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
  document.getElementById("upcoming-predictions").innerHTML = loadingHtml("Predicciones pendientes");
  document.getElementById("upcoming-report").innerHTML = loadingHtml("Reporte pendiente");
  renderWorldcupJobProgress("upcoming-report");
  document.getElementById("match-simulation-grid").innerHTML = loadingHtml("Monte Carlo pendiente");
  document.getElementById("match-simulation-table").innerHTML = "";
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
    "sim-poisson-recent-matches": config.poisson_recent_matches,
    "upcoming-poisson-recent-matches": config.poisson_recent_matches,
    "sim-score-model": config.score_model,
  };
  Object.entries(pairs).forEach(([id, value]) => {
    const input = document.getElementById(id);
    if (input && value !== undefined) input.value = value;
  });
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
  renderHeroHardware(overview.hardware || {});
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
  const matchLine = highlight && highlight.match ? highlight.match : "Horario por confirmar";
  const kickoffLine = [highlight && highlight.date, highlight && highlight.time, highlight && highlight.venue].filter(Boolean).join(" · ");
  container.innerHTML = dashboardCountdownHtml(matchLine, kickoffLine, targetIso, stateLabel);
  refreshCountdowns();
}

function dashboardCountdownHtml(matchLine, kickoffLine, targetIso, stateLabel) {
  return `
    <div class="countdown-head">
      <span data-countdown-label>Próximo</span>
      <strong id="hero-countdown-vs">${escapeHtml(matchLine || "Partido pendiente")}</strong>
      <small>${escapeHtml(kickoffLine || "Horario pendiente")}</small>
    </div>
    <div class="hero-countdown" data-countdown-mode="dashboard" data-countdown-target="${escapeAttr(targetIso || "")}" data-countdown-state="${escapeAttr(stateLabel || "")}"></div>`;
}

function fixtureCountdownHtml(fixture) {
  const kickoff = fixture && fixture.kickoff_iso ? fixture.kickoff_iso : "";
  const countdownState = fixture && fixture.countdown_state ? fixture.countdown_state : "";
  return `<div class="fixture-countdown" data-countdown-mode="fixture" data-countdown-target="${escapeAttr(kickoff)}" data-countdown-state="${escapeAttr(countdownState)}"></div>`;
}

function refreshCountdowns() {
  updateCountdownElements();
  if (!state.countdownTimer) {
    state.countdownTimer = window.setInterval(updateCountdownElements, 1000);
  }
}

function updateCountdownElements() {
  document.querySelectorAll("[data-countdown-target]").forEach((node) => {
    updateCountdownElement(node);
  });
}

function updateCountdownElement(node) {
  const targetIso = node.dataset.countdownTarget || "";
  const mode = node.dataset.countdownMode || "fixture";
  const stateLabel = node.dataset.countdownState || "";
  const target = targetIso ? Date.parse(targetIso) : NaN;
  const status = countdownStatus(target, stateLabel);
  const labelNode = mode === "dashboard" ? node.parentElement && node.parentElement.querySelector("[data-countdown-label]") : null;
  if (labelNode) labelNode.textContent = status.label;
  if (!status.remaining) {
    node.innerHTML = mode === "dashboard"
      ? `<div class="countdown-chip live"><span>Estado</span><strong>${escapeHtml(status.label)}</strong></div>`
      : `<span class="fixture-countdown-state">${escapeHtml(status.label)}</span>`;
    return;
  }
  if (mode === "dashboard") {
    node.innerHTML = [
      countdownChip("Días", status.remaining.days),
      countdownChip("Horas", status.remaining.hours),
      countdownChip("Min", status.remaining.minutes),
      countdownChip("Seg", status.remaining.seconds),
    ].join("");
    return;
  }
  node.innerHTML = [
    fixtureCountdownCell("Días", status.remaining.days),
    fixtureCountdownCell("Horas", status.remaining.hours),
    fixtureCountdownCell("Min", status.remaining.minutes),
    fixtureCountdownCell("Seg", status.remaining.seconds),
  ].join("");
}

function countdownStatus(target, stateLabel) {
  if (stateLabel === "finished") return { label: "Finalizado", remaining: null };
  if (Number.isNaN(target)) return { label: "Horario pendiente", remaining: null };
  const diff = target - Date.now();
  if (diff <= 0) {
    if (stateLabel === "live" || Math.abs(diff) <= 3 * 60 * 60 * 1000) return { label: "En curso", remaining: null };
    return { label: "Finalizado", remaining: null };
  }
  return { label: "Próximo", remaining: countdownParts(diff) };
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

function fixtureCountdownCell(label, value) {
  return `<span><b>${escapeHtml(value)}</b><small>${escapeHtml(label)}</small></span>`;
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

async function runUpcomingPredictions() {
  clearAlert();
  const limit = Number(document.getElementById("upcoming-predict-limit").value || 8);
  const group = document.getElementById("upcoming-group-filter").value || "";
  const pipelineMode = syncUpcomingPipelineControls();
  const sotaCalculationMode = (document.getElementById("upcoming-sota-calculation-mode") || {}).value || "exact";
  const benchmarkTuningEnabled = pipelineMode === "alternatives_benchmark" && Boolean((document.getElementById("upcoming-benchmark-tuning-enabled") || {}).checked);
  const calculationLabel = pipelineMode === "alternatives_benchmark"
    ? `Benchmark alternativas${benchmarkTuningEnabled ? " + Optuna" : ""}`
    : sotaCalculationMode === "monte_carlo"
    ? `SOTA Monte Carlo consenso N=${formatInteger(currentMonteCarloSimulations())}`
    : "Consenso exacto";
  document.getElementById("upcoming-summary").textContent = `Generando ${calculationLabel} con Poisson ultimos ${currentPoissonRecentMatches()}...`;
  try {
    const job = await api("/api/mundial/predict-upcoming-report", jsonOptions({
      ...simulationPayload({
        score_model: "independent_poisson",
      }),
      pipeline_mode: pipelineMode,
      limit,
      group,
      bayes_profile: (document.getElementById("upcoming-bayes-profile") || {}).value || "deep",
      sota_device: (document.getElementById("upcoming-sota-device") || {}).value || "auto",
      sota_calculation_mode: sotaCalculationMode,
      benchmark_tuning_enabled: benchmarkTuningEnabled,
      benchmark_tuning_trials: Number((document.getElementById("upcoming-benchmark-tuning-trials") || {}).value || 20),
      benchmark_tuning_sampler: (document.getElementById("upcoming-benchmark-tuning-sampler") || {}).value || "tpe",
    }));
    trackWorldcupJob(job, "upcoming-report");
  } catch (error) {
    document.getElementById("upcoming-report").innerHTML = loadingHtml("Reporte no disponible");
    showError(error.message);
  }
}

function syncUpcomingPipelineControls() {
  const select = document.getElementById("upcoming-pipeline-mode");
  const mode = (select && select.value) || "poisson_sota";
  const calculation = document.getElementById("upcoming-sota-calculation-mode");
  if (calculation) calculation.disabled = mode === "alternatives_benchmark";
  const sotaControls = document.getElementById("upcoming-sota-controls");
  if (sotaControls) sotaControls.classList.toggle("hidden", mode === "alternatives_benchmark");
  const benchmarkControls = document.getElementById("upcoming-benchmark-controls");
  if (benchmarkControls) benchmarkControls.classList.toggle("hidden", mode !== "alternatives_benchmark");
  return mode;
}

function renderUpcomingReport(report) {
  state.lastUpcomingReport = report;
  const summary = report.summary || {};
  if (summary.pipeline_mode === "alternatives_benchmark") {
    renderAlternativesBenchmarkReport(report);
    return;
  }
  const fixtures = report.fixture_reports || [];
  const hardware = summary.hardware || {};
  const warnings = summary.warnings || [];
  const calculationLabel = summary.sota_calculation_label || (summary.pipeline_mode === "poisson_sota" ? "Consenso exacto: matriz promedio, sin simulacion" : "");
  document.getElementById("upcoming-summary").textContent =
    `${summary.pipeline_label || "Reporte"} - ${summary.returned || 0}/${summary.requested || 0} partidos - ${summary.group || "Todos"} - Poisson ultimos ${summary.poisson_recent_matches || currentPoissonRecentMatches()}${calculationLabel ? ` - ${calculationLabel}` : ""} - ${summary.report_id || report.report_id || ""}`;
  document.getElementById("upcoming-predictions").innerHTML = "";
  document.getElementById("upcoming-report").innerHTML = `
    <div class="report-summary-grid">
      ${reportSummaryCard("Pipeline", summary.pipeline_label || summary.pipeline_mode || "-")}
      ${calculationLabel ? reportSummaryCard("Cálculo", calculationLabel) : ""}
      ${reportSummaryCard("Fuerza global", globalConsensusStrength(fixtures))}
      ${reportSummaryCard("Partidos", `${summary.returned || 0}/${summary.requested || 0}`)}
      ${reportSummaryCard("Hardware", `${hardware.actual_device || "cpu"} · ${hardware.requested_device || "auto"}`)}
      ${reportSummaryCard("Guardado", report.report_path || "latest.json")}
    </div>
    ${reportDownloadButtonsHtml(report.downloads || {}, false)}
    ${warnings.length ? `<div class="warning-list">${warnings.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
    ${clientReportHtml(fixtures)}
    <details class="technical-report-drawer">
      <summary>Vista técnica completa</summary>
      <div class="upcoming-grid">
        ${fixtures.map((fixtureReport) => reportFixtureCardHtml(fixtureReport)).join("") || loadingHtml("Sin fixtures futuros")}
      </div>
    </details>`;
  refreshCountdowns();
  renderTable("upcoming-predictions-table", report.table);
}

function renderAlternativesBenchmarkReport(report) {
  const summary = report.summary || {};
  const fixtures = report.fixture_reports || [];
  const backtests = report.model_backtests || [];
  const best = report.best_model || summary.best_model || {};
  const warnings = summary.warnings || [];
  const tuning = summary.benchmark_tuning || {};
  const backtestAutoN = summary.backtest_auto_n ?? (summary.backtest || {}).evaluated_matches ?? 0;
  const resultSource = summary.backtest_source || summary.result_source || ((summary.results_refresh || {}).source) || "CSV local";
  const range = summary.backtest_range || (summary.backtest || {}).backtest_range || {};
  const firstMatch = range.first_match || {};
  const lastMatch = range.last_match || {};
  document.getElementById("upcoming-summary").textContent =
    `${summary.pipeline_label || "Benchmark alternativas"} - ${fixtures.length}/${summary.requested || 0} próximos - ${backtests.length} modelos - ${backtestAutoN} finalizados detectados - Poisson ultimos ${summary.poisson_recent_matches || currentPoissonRecentMatches()} - ${summary.report_id || report.report_id || ""}`;
  document.getElementById("upcoming-predictions").innerHTML = "";
  document.getElementById("upcoming-report").innerHTML = `
    <div class="report-summary-grid">
      ${reportSummaryCard("Modelo #1", best.available ? (best.model_label || best.model_key || "-") : "-")}
      ${reportSummaryCard("Evaluados", `${backtestAutoN} partidos`)}
      ${reportSummaryCard("Partidos próximos", `${fixtures.length}/${summary.requested || 0}`)}
      ${reportSummaryCard("Criterio", "Score de resultados")}
      ${reportSummaryCard("Primer evaluado", firstMatch.match || `${firstMatch.home || "-"} vs ${firstMatch.away || "-"}`)}
      ${reportSummaryCard("Último evaluado", lastMatch.match || `${lastMatch.home || "-"} vs ${lastMatch.away || "-"}`)}
      ${reportSummaryCard("Generado", formatReportDateTime(summary.generated_at || range.generated_at || report.created_at))}
      ${reportSummaryCard("Fuente resultados", resultSource)}
      ${tuning.enabled ? reportSummaryCard("Optuna N", tuning.available ? `${tuning.best_poisson_recent_matches} ultimos` : "No disponible") : ""}
    </div>
    ${reportDownloadButtonsHtml(report.downloads || {}, true)}
    ${warnings.length ? `<div class="warning-list">${warnings.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>` : ""}
    ${alternativesBenchmarkHtml(report)}`;
  refreshCountdowns();
  renderTable("upcoming-predictions-table", report.table);
}

function alternativesBenchmarkHtml(report) {
  const summary = report.summary || {};
  const fixtures = report.fixture_reports || [];
  const rankedModels = report.ranked_models || report.alternatives || [];
  const backtests = report.model_backtests || [];
  const best = report.best_model || summary.best_model || {};
  const backtestAutoN = summary.backtest_auto_n ?? (summary.backtest || {}).evaluated_matches ?? 0;
  const range = summary.backtest_range || (summary.backtest || {}).backtest_range || {};
  const firstMatch = range.first_match || {};
  const lastMatch = range.last_match || {};
  const tuning = summary.benchmark_tuning || {};
  const tuningLabel = tuning.enabled && tuning.available ? ` · Optuna N=${tuning.best_poisson_recent_matches}` : "";
  return `<section class="client-report-shell">
    <header>
      <div>
        <h3>Score de resultados 2026</h3>
        <small>${escapeHtml(backtestAutoN)} evaluados · ${escapeHtml(firstMatch.home || "-")} vs ${escapeHtml(firstMatch.away || "-")} a ${escapeHtml(lastMatch.home || "-")} vs ${escapeHtml(lastMatch.away || "-")} · generado ${escapeHtml(formatReportDateTime(range.generated_at || summary.generated_at))}${escapeHtml(tuningLabel)}</small>
      </div>
      <span>${escapeHtml(rankedModels.length || backtests.length)} modelo${(rankedModels.length || backtests.length) === 1 ? "" : "s"}</span>
    </header>
    ${bestAlternativeHtml(best)}
    ${featureResearchHtml(summary.feature_research || report.feature_research || {})}
    ${benchmarkTuningHtml(tuning)}
    ${backtestTableHtml(backtests, summary.backtest || {})}
    ${backtestPredictionReviewHtml(backtests)}
    <section class="report-panel">
      <header><strong>Predicciones futuras</strong><small>${escapeHtml(fixtures.length)} fixture${fixtures.length === 1 ? "" : "s"}</small></header>
      <div class="client-report-grid alternatives-fixture-grid">
        ${fixtures.map((fixtureReport) => alternativeFixtureCardHtml(fixtureReport)).join("") || loadingHtml("Sin fixtures futuros")}
      </div>
    </section>
    <section class="report-panel">
      <header><strong>Detalles por modelo</strong><small>Colapsables para revisar disponibilidad, métricas y advertencias</small></header>
      <div class="alternatives-model-list">
        ${rankedModels.map((item) => alternativeBenchmarkCardHtml(item)).join("") || loadingHtml("Sin modelos")}
      </div>
    </section>
  </section>`;
}

function bestAlternativeHtml(best) {
  const item = best || {};
  if (!item.available) {
    return `<section class="report-panel">
      <header><strong>Modelo #1</strong><small>No disponible</small></header>
      <p>${escapeHtml(item.reason || "El backtest no devolvio un modelo evaluable.")}</p>
    </section>`;
  }
  return `<section class="report-panel best-alternative-panel">
    <header><strong>Modelo #1</strong><small>${escapeHtml(item.selection_policy || item.ranking_reason || "")}</small></header>
    <div class="client-main-pick">
      <span>Score de resultados ${escapeHtml(formatNumber(item.score_resultados ?? item.reliability_score ?? 0))}</span>
      <strong>${escapeHtml(item.model_label || item.model_key || "")}</strong>
      <small>${escapeHtml(item.evaluated_matches || 0)} finalizados detectados: ${escapeHtml(item.holdout_start || "")} a ${escapeHtml(item.holdout_end || "")}</small>
    </div>
    <div class="technical-meta-row">
      <span>Pick ${escapeHtml(formatNumber(Number(item.pick_accuracy || 0) * 100))}%</span>
      <span>Marcador #1 ${escapeHtml(formatNumber(Number(item.score_accuracy || 0) * 100))}%</span>
      <span>Top-3 marcador ${escapeHtml(formatNumber(Number(item.top3_score_accuracy || 0) * 100))}%</span>
      <span>U/O ${escapeHtml(formatNumber(Number(item.over_under_accuracy || 0) * 100))}%</span>
      <span>Calibración ${escapeHtml(formatNumber(Math.max(0, 100 - Number(item.expected_calibration_error || 0) * 100)))}%</span>
    </div>
  </section>`;
}

function featureResearchHtml(featureResearch) {
  const item = featureResearch || {};
  const families = item.families || [];
  const basis = item.research_basis || [];
  if (!families.length) return "";
  const active = item.active_or_cached_families || [];
  return `<details class="report-panel feature-research-panel">
    <summary><strong>Features para mejorar modelos</strong><small>${escapeHtml(active.length)} familias activas/cacheadas · sin leakage temporal</small></summary>
    <div class="technical-meta-row">
      <span>${escapeHtml(item.anti_leakage || "Corte temporal antes del partido")}</span>
      <span>${escapeHtml(item.recommendation || "")}</span>
    </div>
    <div class="client-report-grid alternatives-model-grid">
      ${families.map((family) => `<article class="benchmark-metric-card">
        <header>
          <span>${escapeHtml(family.status || "")}</span>
          <strong>${escapeHtml(family.label || family.key || "")}</strong>
        </header>
        <small>${escapeHtml(family.impact || "")}</small>
        <div class="technical-meta-row compact">
          ${(family.features || []).slice(0, 4).map((feature) => `<span>${escapeHtml(feature)}</span>`).join("")}
        </div>
      </article>`).join("")}
    </div>
    ${basis.length ? `<div class="technical-meta-row research-basis-row">
      ${basis.map((source) => {
        const title = typeof source === "string" ? source : source.title || "";
        const label = typeof source === "string" ? source : `${source.title || ""}: ${source.apply || source.finding || ""}`;
        const url = typeof source === "string" ? "" : source.url || "";
        return url
          ? `<a href="${escapeAttr(url)}" target="_blank" rel="noopener" title="${escapeAttr((source.finding || "").trim())}">${escapeHtml(label)}</a>`
          : `<span>${escapeHtml(title || label)}</span>`;
      }).join("")}
    </div>` : ""}
  </details>`;
}

function benchmarkTuningHtml(tuning) {
  const item = tuning || {};
  if (!item.enabled) return "";
  const warnings = item.warnings || [];
  return `<section class="report-panel">
    <header><strong>Optuna N ultimos</strong><small>${escapeHtml(item.sampler || "tpe")} · ${escapeHtml(item.objective || "mean_score_resultados")}</small></header>
    <div class="technical-meta-row">
      <span>N ${escapeHtml(item.available ? item.best_poisson_recent_matches : "-")}</span>
      <span>Trials ${escapeHtml(item.n_trials || 0)}</span>
      <span>Score ${escapeHtml(item.best_value !== null && item.best_value !== undefined ? formatNumber(item.best_value) : "-")}</span>
      <span>${escapeHtml(item.scope === "all_active_models" ? "Todos los modelos activos" : item.scope || "")}</span>
      <span>${escapeHtml(item.available ? "Aplicado" : "No disponible")}</span>
    </div>
    ${warnings.length ? `<div class="warning-list">${warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>` : ""}
  </section>`;
}

function alternativeFixtureCardHtml(report) {
  const fixture = report.fixture || {};
  const homeAsset = fixture.home_asset || assetFor(fixture.home || "");
  const awayAsset = fixture.away_asset || assetFor(fixture.away || "");
  const leader = report.primary_model || {};
  return `<article class="client-fixture-card alternative-fixture-card">
    <header>
      <span>${escapeHtml([fixture.date || "", fixture.time || ""].filter(Boolean).join(" · "))}</span>
      <strong>${escapeHtml(fixture.group || "")}</strong>
    </header>
    ${fixtureCountdownHtml(fixture)}
    <div class="client-match-row">
      <div>${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
      <b>vs</b>
      <div>${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
    </div>
    ${topRankedFixturePickHtml(leader, fixture)}
  </article>`;
}

function topRankedFixturePickHtml(model, fixture) {
  const item = model || {};
  if (!item.model_key) return "";
  const probs = item.probabilities || {};
  const topScores = item.top_scores || [];
  const decision = item.decision || {};
  const activeOutcome = decision.outcome || strongestOutcomeFromProbabilities(probs);
  const pickLabel = decision.label || outcomeLabel(activeOutcome) || "-";
  const pickTeam = activeOutcome === "draw" ? "Empate" : activeOutcome === "home" ? (fixture || {}).home : activeOutcome === "away" ? (fixture || {}).away : "-";
  const confidence = Math.max(Number(probs.home || 0), Number(probs.draw || 0), Number(probs.away || 0));
  const topScore = topScores[0] || {};
  if (item.available === false) {
    return `<section class="future-prediction-panel top-ranked-pick unavailable">
      <header><strong>Distribuciones</strong><small>${escapeHtml(item.reason || "Modelo no disponible para este fixture")}</small></header>
    </section>`;
  }
  return `<section class="future-prediction-panel top-ranked-pick">
    <header><strong>Pronóstico visual</strong><small>${escapeHtml(item.model_label || item.model_key || "Modelo #1")}</small></header>
    <div class="future-pick-ribbon">
      <span>Pick 1X2</span>
      <strong>${escapeHtml(pickLabel)} · ${escapeHtml(pickTeam || "-")}</strong>
      <small>${escapeHtml(formatProbability(confidence))}% · marcador #1 ${escapeHtml(topScore.score || "-")}</small>
    </div>
    ${modelOutcomeProbabilitiesHtml(probs, activeOutcome, fixture)}
    ${modelOverUnderProbabilitiesHtml(probs, item.totals || {})}
    ${futureTopScoresHtml(topScores)}
  </section>`;
}

function modelOutcomeProbabilitiesHtml(probabilities, activeOutcome, fixture) {
  const probs = probabilities || {};
  const labels = [
    { key: "home", label: "1", team: (fixture || {}).home || "Local" },
    { key: "draw", label: "X", team: "Empate" },
    { key: "away", label: "2", team: (fixture || {}).away || "Visitante" },
  ];
  return `<div class="future-outcome-bars model-outcome-probabilities" aria-label="Probabilidades 1X2 por modelo">
    ${labels.map((item) => {
      const value = probs[item.key];
      const active = activeOutcome === item.key;
      return `<div class="${escapeAttr(active ? "active" : "")}">
        <span>${escapeHtml(item.label)}</span>
        <strong>${escapeHtml(formatProbability(value))}%</strong>
        <small>${escapeHtml(item.team)}</small>
        <i><b style="width:${escapeAttr(clampPercent(value))}%"></b></i>
      </div>`;
    }).join("")}
  </div>`;
}

function modelOverUnderProbabilitiesHtml(probabilities, totals) {
  const probs = probabilities || {};
  const picks = totals || {};
  return `<div class="future-total-cards model-total-probabilities" aria-label="Over under por modelo">
    ${goalMarketLines.map((line) => {
      const lineKey = line.label.replace("U/O ", "");
      const over = Number(probs[line.over]);
      const under = Number(probs[line.under]);
      const pick = picks[lineKey] || (over >= under ? "over" : "under");
      const pickLabel = pick === "over" ? "Over" : "Under";
      const overWidth = clampPercent(over);
      const underWidth = clampPercent(under);
      return `<div class="${escapeAttr(pick)}">
        <header><span>${escapeHtml(line.label)}</span><strong>${escapeHtml(pickLabel)}</strong></header>
        <i><b style="width:${escapeAttr(overWidth)}%"></b><em style="width:${escapeAttr(underWidth)}%"></em></i>
        <small><b>O ${escapeHtml(formatProbability(over))}%</b><b>U ${escapeHtml(formatProbability(under))}%</b></small>
      </div>`;
    }).join("")}
  </div>`;
}

function futureTopScoresHtml(topScores) {
  const scores = topScores || [];
  if (!scores.length) return "";
  return `<div class="future-score-strip" aria-label="Marcadores más probables">
    ${scores.slice(0, 5).map((score, index) => `<span class="${escapeAttr(index === 0 ? "primary" : "")}"><small>#${escapeHtml(index + 1)}</small><strong>${escapeHtml(score.score || "-")}</strong><b>${escapeHtml(formatNumber(score.probability ?? 0))}%</b></span>`).join("")}
  </div>`;
}

function formatProbability(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return formatNumber(number);
}

function backtestTableHtml(backtests, summary) {
  const rows = backtests || [];
  return `<details class="report-panel alternatives-backtest-panel">
    <summary>
      <strong>Métricas completas</strong>
      <small>${escapeHtml((summary || {}).evaluated_matches || 0)} evaluados 2026 · ${escapeHtml((summary || {}).train_matches || 0)} históricos base · ${escapeHtml((summary || {}).source || "")}</small>
    </summary>
    <div class="benchmark-bar-list">
      ${rows.map((item) => benchmarkMetricCardHtml(item)).join("") || loadingHtml("Sin backtest")}
    </div>
  </details>`;
}

function benchmarkMetricCardHtml(item) {
  const vs = item.vs_poisson || {};
  const score = Number(item.score_resultados ?? item.reliability_score ?? 0);
  const badgeClass = score >= 75 ? "high" : score >= 45 ? "medium" : "low";
  return `<article class="benchmark-metric-card">
    <header>
      <span>#${escapeHtml(item.rank || "-")}</span>
      <strong>${escapeHtml(item.model_label || item.model_key || "")}</strong>
      <b class="reliability-badge ${escapeAttr(badgeClass)}">${escapeHtml(formatNumber(score))}</b>
    </header>
    <div class="metric-bar-grid">
      ${compactMetricBarHtml("Score resultados", formatNumber(score), score)}
      ${compactMetricBarHtml("Log-loss", formatNumber(item.log_loss ?? "-"), inverseMetricPercent(item.log_loss, 1.6))}
      ${compactMetricBarHtml("RPS", formatNumber(item.rps ?? "-"), inverseMetricPercent(item.rps, 0.75))}
      ${compactMetricBarHtml("ECE", formatNumber(item.expected_calibration_error ?? "-"), inverseMetricPercent(item.expected_calibration_error, 0.35))}
      ${compactMetricBarHtml("Pick %", `${formatNumber(Number(item.pick_accuracy || 0) * 100)}%`, Number(item.pick_accuracy || 0) * 100)}
      ${compactMetricBarHtml("Marcador #1", `${formatNumber(Number(item.score_accuracy || 0) * 100)}%`, Number(item.score_accuracy || 0) * 100)}
      ${compactMetricBarHtml("Top-3 marcador", `${formatNumber(Number(item.top3_score_accuracy || 0) * 100)}%`, Number(item.top3_score_accuracy || 0) * 100)}
      ${compactMetricBarHtml("Brier", formatNumber(item.brier ?? "-"), inverseMetricPercent(item.brier, 0.75))}
      ${compactMetricBarHtml("U/O 2.5 LL", formatNumber(item.ou25_log_loss ?? "-"), inverseMetricPercent(item.ou25_log_loss, 1.2))}
      ${compactMetricBarHtml("Vs Poisson", `${escapeHtml(vs.metric_wins ?? 0)}/${escapeHtml(vs.metric_total ?? 7)}`, Number(vs.metric_wins || 0) * (100 / Math.max(Number(vs.metric_total || 7), 1)))}
    </div>
    <small>${escapeHtml(vs.summary || "")}</small>
  </article>`;
}

function compactMetricBarHtml(label, value, percent) {
  return `<div class="compact-metric-bar">
    <span>${escapeHtml(label)}</span>
    <i><b style="width:${escapeAttr(clampPercent(percent))}%"></b></i>
    <strong>${escapeHtml(value)}</strong>
  </div>`;
}

function inverseMetricPercent(value, ceiling) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((1 - Math.min(number / ceiling, 1)) * 100)));
}

function backtestPredictionReviewHtml(backtests) {
  const leader = (backtests || []).find((item) => item && item.available) || (backtests || [])[0] || {};
  const rows = leader.matches || leader.sample || [];
  if (!rows.length) return "";
  return `<section class="report-panel backtest-review-panel">
    <header>
      <strong>Backtest: predicción vs resultado</strong>
      <small>${escapeHtml(leader.model_label || leader.model_key || "Modelo #1")} · ${escapeHtml(rows.length)} partidos evaluados</small>
    </header>
    <div class="backtest-match-card-grid">
      ${rows.map((row) => backtestMatchCardHtml(row)).join("")}
    </div>
    <details class="backtest-full-table-drawer">
      <summary>Tabla completa</summary>
      ${backtestMatchTableHtml(leader)}
    </details>
  </section>`;
}

function backtestMatchCardHtml(row) {
  const item = row || {};
  const teams = backtestMatchTeams(item);
  const homeAsset = item.home_asset || assetFor(teams.home || "");
  const awayAsset = item.away_asset || assetFor(teams.away || "");
  return `<article class="backtest-match-card">
    <header>
      <span>${escapeHtml(item.date || "")}</span>
      <strong>${escapeHtml(item.fixture_id ? `#${item.fixture_id}` : "")}</strong>
    </header>
    <div class="backtest-match-teams">
      <div>${flagHtml(homeAsset)}<strong>${escapeHtml(teams.home || "")}</strong></div>
      <b>${escapeHtml(item.actual_score || "-")}</b>
      <div>${flagHtml(awayAsset)}<strong>${escapeHtml(teams.away || "")}</strong></div>
    </div>
    <div class="backtest-card-metrics">
      <section>
        <span>Pick 1X2</span>
        ${backtestOutcomeCellHtml(item)}
      </section>
      <section>
        <span>Marcador #1</span>
        ${backtestScoreCellHtml(item)}
      </section>
      <section>
        <span>Top-3 / Top-5</span>
        <div class="backtest-top-status">
          <b class="${escapeAttr(item.top3_score_hit ? "hit" : "miss")}">Top-3 ${escapeHtml(item.top3_score_hit ? "Si" : "No")}</b>
          <b class="${escapeAttr(item.top5_score_hit ? "hit" : "miss")}">Top-5 ${escapeHtml(item.top5_score_hit ? "Si" : "No")}</b>
        </div>
      </section>
    </div>
    ${backtestTopScoresHtml(item)}
    ${backtestOverUnderSummaryHtml(item.over_under || [])}
  </article>`;
}

function backtestMatchTeams(row) {
  const item = row || {};
  if (item.home || item.away) return { home: item.home || "", away: item.away || "" };
  const parts = String(item.match || "").split(" vs ");
  return { home: parts[0] || "", away: parts[1] || "" };
}

function backtestTopScoresHtml(row) {
  const scores = (row.top_scores || []).length
    ? row.top_scores
    : [{ rank: 1, score: row.most_probable_score || row.modal_score || "-", probability: row.most_probable_score_probability, hit: row.most_probable_score_hit || row.score_hit }];
  return `<div class="backtest-top-scores" aria-label="Top marcadores">
    ${scores.slice(0, 5).map((score) => `<span class="${escapeAttr(score.hit ? "hit" : "")}"><b>#${escapeHtml(score.rank || "")} ${escapeHtml(score.score || "-")}</b><small>${escapeHtml(formatNumber(score.probability ?? "-"))}%</small></span>`).join("")}
  </div>`;
}

function backtestOverUnderSummaryHtml(rows) {
  const items = ["0.5", "1.5", "2.5", "3.5"].map((line) => (rows || []).find((item) => String(item.line) === line)).filter(Boolean);
  if (!items.length) return "";
  return `<div class="backtest-ou-summary">
    ${items.map((item) => `<span class="${escapeAttr(item.hit ? "hit" : "miss")}"><b>U/O ${escapeHtml(item.line || "")}</b><small>${escapeHtml(item.prediction_label || "-")} · real ${escapeHtml(item.actual_label || "-")}</small></span>`).join("")}
  </div>`;
}

function backtestMatchTableHtml(backtest) {
  const rows = (backtest && (backtest.matches || backtest.sample)) || [];
  if (!rows.length) return loadingHtml("Sin partidos evaluados");
  return `<div class="alternatives-backtest-table backtest-match-table">
    <table>
      <thead><tr><th>Fecha</th><th>Partido</th><th>Resultado</th><th>Marcador #1</th><th>1X2 pred.</th><th>1X2 real</th><th>U/O 0.5</th><th>U/O 1.5</th><th>U/O 2.5</th><th>U/O 3.5</th></tr></thead>
      <tbody>${rows.map((row) => `
        <tr>
          <td>${escapeHtml(row.date || "")}</td>
          <td>${escapeHtml(row.match || "")}</td>
          <td>${escapeHtml(row.actual_score || "")}</td>
          <td>${backtestScoreCellHtml(row)}</td>
          <td>${backtestOutcomeCellHtml(row)}</td>
          <td>${escapeHtml(row.actual_pick || "-")}</td>
          ${["0.5", "1.5", "2.5", "3.5"].map((line) => `<td>${backtestOverUnderCellHtml((row.over_under || []).find((item) => String(item.line) === line))}</td>`).join("")}
        </tr>
      `).join("")}</tbody>
    </table>
  </div>`;
}

function backtestScoreCellHtml(row) {
  const score = (row && (row.most_probable_score || row.modal_score)) || "-";
  const hit = Boolean(row && (row.most_probable_score_hit || row.score_hit));
  const probability = Number(row && row.most_probable_score_probability);
  const top3 = row && row.top3_score_hit ? " · Top-3" : "";
  const probabilityText = Number.isFinite(probability) ? `${formatNumber(probability)}%` : "-";
  return `<span class="backtest-result-cell ${escapeAttr(hit ? "hit" : "miss")}"><b>${escapeHtml(score)}</b><small>${escapeHtml(hit ? "Acertó" : "Falló")} · ${escapeHtml(probabilityText)}${escapeHtml(top3)}</small></span>`;
}

function backtestOutcomeCellHtml(row) {
  const hit = row && row.pick_hit;
  const probability = Number(row && row.actual_probability);
  const probabilityText = Number.isFinite(probability) ? ` · real ${formatNumber(probability)}%` : "";
  return `<span class="backtest-result-cell ${escapeAttr(hit ? "hit" : "miss")}"><b>${escapeHtml((row && row.pick) || "-")}</b><small>${escapeHtml(hit ? "Acertó" : "Falló")}${escapeHtml(probabilityText)}</small></span>`;
}

function backtestOverUnderCellHtml(item) {
  if (!item) return "-";
  const hit = item.hit;
  return `<span class="backtest-result-cell ${escapeAttr(hit ? "hit" : "miss")}"><b>${escapeHtml(item.prediction_label || "-")} ${escapeHtml(item.line || "")}</b><small>Real ${escapeHtml(item.actual_label || "-")} · ${escapeHtml(formatNumber(item.confidence ?? 0))}%</small></span>`;
}

function alternativeBenchmarkCardHtml(item) {
  const backtest = item.backtest || {};
  const vs = backtest.vs_poisson || {};
  const warnings = backtest.warnings || [];
  const score = Number(backtest.score_resultados ?? backtest.reliability_score ?? 0);
  const badgeClass = score >= 75 ? "high" : score >= 45 ? "medium" : "low";
  return `<details class="alternative-model-card">
    <summary>
      <span>#${escapeHtml(backtest.rank || item.rank || "")}</span>
      <strong>${escapeHtml(item.model_name || item.key || "")}</strong>
      <b class="reliability-badge ${escapeAttr(badgeClass)}">${escapeHtml(formatNumber(score))}</b>
    </summary>
    <div class="client-main-pick">
      <span>Modelo</span>
      <strong>${escapeHtml(item.model_name || item.key || "")}</strong>
      <small>${escapeHtml(item.description || "")}</small>
    </div>
    <section class="client-bets-panel">
      <header><strong>Backtest automático ${escapeHtml(backtest.evaluated_matches || 0)}</strong><small>${escapeHtml(vs.beats_poisson ? "Mejora vs Poisson" : "Métricas")}</small></header>
      <div class="technical-meta-row">
        <span>Score resultados ${escapeHtml(formatNumber(score))}</span>
        <span>Log-loss ${escapeHtml(formatNumber(backtest.log_loss ?? "-"))}</span>
        <span>RPS ${escapeHtml(formatNumber(backtest.rps ?? "-"))}</span>
        <span>ECE ${escapeHtml(formatNumber(backtest.expected_calibration_error ?? "-"))}</span>
        <span>Marcador #1 ${escapeHtml(formatNumber(Number(backtest.score_accuracy || 0) * 100))}%</span>
        <span>Top-3 ${escapeHtml(formatNumber(Number(backtest.top3_score_accuracy || 0) * 100))}%</span>
        <span>Brier ${escapeHtml(formatNumber(backtest.brier ?? "-"))}</span>
        <span>Pick ${escapeHtml(formatNumber(Number(backtest.pick_accuracy || 0) * 100))}%</span>
        <span>U/O2.5 LL ${escapeHtml(formatNumber(backtest.ou25_log_loss ?? "-"))}</span>
      </div>
    </section>
    ${backtestMatchTableHtml(backtest)}
    ${warnings.length ? `<div class="warning-list compact">${warnings.map((warning) => `<span>${escapeHtml(warning)}</span>`).join("")}</div>` : ""}
  </details>`;
}

function reportSummaryCard(label, value) {
  return `<article class="report-summary-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>`;
}

function reportDownloadButtonsHtml(downloads, includeBacktest) {
  const items = [
    { href: downloads.predictions_html, label: "Predicciones HTML" },
    { href: downloads.predictions_csv, label: "Predicciones CSV" },
    includeBacktest ? { href: downloads.backtest_html, label: "Backtesting HTML" } : null,
    includeBacktest ? { href: downloads.backtest_csv, label: "Backtesting CSV" } : null,
  ].filter((item) => item && item.href);
  if (!items.length) return "";
  return `<div class="report-download-actions">
    ${items.map((item) => `<a href="${escapeAttr(item.href)}" target="_blank" rel="noopener">${escapeHtml(item.label)}</a>`).join("")}
  </div>`;
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

function clientReportHtml(fixtures) {
  const items = fixtures || [];
  return `<section class="client-report-shell">
    <header>
      <div>
        <h3>Reporte cliente</h3>
        <small>Resultados listos para apuestas, sin detalle técnico de modelos</small>
      </div>
      <span>${escapeHtml(items.length)} partido${items.length === 1 ? "" : "s"}</span>
    </header>
    <div class="client-report-grid">
      ${items.map((fixtureReport) => clientFixtureCardHtml(fixtureReport)).join("") || loadingHtml("Sin resultados para cliente")}
    </div>
  </section>`;
}

function clientFixtureCardHtml(report) {
  const fixture = report.fixture || {};
  const primary = clientPrimaryReportPayload(report);
  const consensus = primary.consensus || {};
  const stats = primary.stats || {};
  const distribution = primary.distribution || {};
  const homeAsset = fixture.home_asset || assetFor(fixture.home || "");
  const awayAsset = fixture.away_asset || assetFor(fixture.away || "");
  const pickTeam = consensus.outcome === "draw" ? "Empate" : consensus.outcome === "home" ? fixture.home : consensus.outcome === "away" ? fixture.away : "-";
  const pickLabel = consensus.outcome_label || "-";
  const pickConfidence = Number.isFinite(Number(primary.pickConfidence)) ? Number(primary.pickConfidence) : clientOutcomeConfidence(stats, consensus);
  const confidenceClass = pickConfidence >= 70 ? "high" : pickConfidence >= 55 ? "medium" : "low";
  return `<article class="client-fixture-card confidence-${escapeAttr(confidenceClass)}">
    <header>
      <span>${escapeHtml([fixture.date || "", fixture.time || ""].filter(Boolean).join(" · "))}</span>
      <strong>${escapeHtml(fixture.group || "")}</strong>
    </header>
    ${fixtureCountdownHtml(fixture)}
    <div class="client-match-row">
      <div>${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
      <b>vs</b>
      <div>${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
    </div>
    <div class="client-main-pick">
      <span>Pronóstico principal</span>
      <strong>${escapeHtml(pickLabel)} · ${escapeHtml(pickTeam || "-")}</strong>
      <small>${escapeHtml(formatNumber(pickConfidence))}% confianza · ${escapeHtml(primary.label || consensus.strength || "Baja")}</small>
    </div>
    ${clientOutcomeChartHtml(stats, consensus, fixture, primary)}
    ${clientTotal25PanelHtml(report, primary)}
    ${clientBetCardsHtml(report, primary)}
    ${clientScorePanelHtml(distribution, primary)}
  </article>`;
}

function clientPrimaryReportPayload(report) {
  const mc = (report && report.monte_carlo_consensus) || {};
  if (mc.available) {
    const probabilities = mc.probabilities || {};
    const outcome = mc.outcome || strongestOutcomeFromProbabilities(probabilities);
    const outcomeProbability = Number(mc.outcome_probability ?? probabilities[outcome] ?? 0);
    return {
      mode: "monte_carlo",
      label: `SOTA Monte Carlo consenso: N=${formatInteger(mc.iterations || 0)}`,
      probabilities,
      distribution: mc,
      stats: report.model_statistics || {},
      pickConfidence: outcomeProbability,
      consensus: {
        ...(report.consensus || {}),
        outcome,
        outcome_label: mc.outcome_label || outcomeLabel(outcome),
        outcome_share: outcomeProbability / 100,
        strength: "Monte Carlo",
        totals: mc.totals || {},
      },
    };
  }
  const consensus = (report && report.consensus) || {};
  return {
    mode: "exact",
    label: (report && report.sota_calculation_label) || "Consenso exacto: matriz promedio, sin simulacion",
    probabilities: {},
    distribution: (report && report.consensus_score_distribution) || {},
    stats: (report && report.model_statistics) || {},
    consensus,
    pickConfidence: null,
  };
}

function strongestOutcomeFromProbabilities(probabilities) {
  const probs = probabilities || {};
  return [
    ["home", Number(probs.home || 0)],
    ["draw", Number(probs.draw || 0)],
    ["away", Number(probs.away || 0)],
  ].sort((a, b) => b[1] - a[1])[0][0];
}

function outcomeLabel(outcome) {
  return { home: "1", draw: "X", away: "2" }[outcome] || "";
}

function clientOutcomeConfidence(stats, consensus) {
  const outcome = (consensus || {}).outcome || "";
  const summary = (((stats || {}).outcomes || {})[outcome]) || {};
  if (Number.isFinite(Number(summary.avg))) return Number(summary.avg);
  return Number((consensus || {}).outcome_share || 0) * 100;
}

function clientOutcomeChartHtml(stats, consensus, fixture, primary) {
  const outcomeStats = (stats && stats.outcomes) || {};
  const outcomeCounts = (consensus && consensus.outcome_counts) || {};
  const eligible = Math.max(Number((stats && stats.model_count) || (consensus && consensus.eligible_models) || 0), 1);
  const directProbabilities = (primary && primary.probabilities) || {};
  const rows = [
    { key: "home", label: "1", team: fixture.home || "Local" },
    { key: "draw", label: "X", team: "Empate" },
    { key: "away", label: "2", team: fixture.away || "Visitante" },
  ];
  return `<div class="client-outcome-chart">
    ${rows.map((item) => {
      const summary = outcomeStats[item.key] || {};
      const direct = Number(directProbabilities[item.key]);
      const value = Number.isFinite(direct) && primary && primary.mode === "monte_carlo"
        ? direct
        : Number.isFinite(Number(summary.avg)) ? Number(summary.avg) : (Number(outcomeCounts[item.key] || 0) / eligible) * 100;
      return `<div class="${escapeAttr(item.key === ((consensus || {}).outcome || "") ? "active" : "")}">
        <span>${escapeHtml(item.label)}</span>
        <strong>${escapeHtml(formatNumber(value))}%</strong>
        <small>${escapeHtml(item.team)}</small>
        <i style="height:${escapeAttr(clampPercent(value))}%"></i>
      </div>`;
    }).join("")}
  </div>`;
}

function clientBetCardsHtml(report, primary) {
  const bets = clientBestBets(report, primary).filter((bet) => bet.market !== "U/O 2.5").slice(0, 3);
  if (!bets.length) return "";
  return `<section class="client-bets-panel">
    <header><strong>Mejores señales</strong><small>Ordenadas por acuerdo</small></header>
    <div class="client-bet-grid">
      ${bets.map((bet) => `<div class="client-bet-card">
        <span>${escapeHtml(bet.market)}</span>
        <strong>${escapeHtml(bet.pick)}</strong>
        <b>${escapeHtml(formatNumber(bet.probability))}%</b>
        <small>${escapeHtml(bet.detail)}</small>
      </div>`).join("")}
    </div>
  </section>`;
}

function clientTotal25PanelHtml(report, primary) {
  const bet = clientBestBets(report, primary).find((item) => item.market === "U/O 2.5");
  if (!bet) return "";
  return `<section class="client-total25-panel">
    <span>U/O 2.5</span>
    <strong>${escapeHtml(bet.pick || "-")} · ${escapeHtml(formatNumber(bet.probability || 0))}%</strong>
    <small>${escapeHtml(bet.detail || "")}</small>
  </section>`;
}

function clientBestBets(report, primary) {
  if (primary && primary.mode === "monte_carlo") {
    return clientMonteCarloBets(primary);
  }
  const fixture = report.fixture || {};
  const consensus = report.consensus || {};
  const stats = report.model_statistics || {};
  const distribution = report.consensus_score_distribution || {};
  const bets = [];
  const pickConfidence = clientOutcomeConfidence(stats, consensus);
  const pickTeam = consensus.outcome === "draw" ? "Empate" : consensus.outcome === "home" ? fixture.home : consensus.outcome === "away" ? fixture.away : "";
  bets.push({
    market: "1X2",
    pick: `${consensus.outcome_label || "-"} ${pickTeam || ""}`.trim(),
    probability: pickConfidence,
    agreement: Number(consensus.outcome_share || 0),
    detail: `${Math.round(Number(consensus.outcome_share || 0) * 100)}% acuerdo`,
  });
  Object.entries((stats && stats.totals) || {}).forEach(([line, item]) => {
    const pick = item.pick || "";
    const label = item.label || (pick === "over" ? "Over" : pick === "under" ? "Under" : "");
    const summary = pick === "over" ? item.over || {} : item.under || {};
    bets.push({
      market: `U/O ${line}`,
      pick: label || "-",
      probability: Number(summary.avg || 0),
      agreement: Number(item.share || 0),
      detail: `${Math.round(Number(item.share || 0) * 100)}% acuerdo · σ ${formatNumber(summary.std || 0)}`,
    });
  });
  const topScore = ((distribution || {}).top_scores || [])[0];
  if (topScore) {
    bets.push({
      market: "Marcador exacto",
      pick: topScore.score || "-",
      probability: Number(topScore.probability || 0),
      agreement: 0,
      detail: "score más probable",
    });
  }
  return bets.sort((a, b) => (Number(b.agreement || 0) - Number(a.agreement || 0)) || (Number(b.probability || 0) - Number(a.probability || 0)));
}

function clientMonteCarloBets(primary) {
  const consensus = primary.consensus || {};
  const distribution = primary.distribution || {};
  const totals = consensus.totals || {};
  const bets = [{
    market: "1X2",
    pick: consensus.outcome_label || "-",
    probability: Number(primary.pickConfidence || 0),
    agreement: Number(primary.pickConfidence || 0) / 100,
    detail: "probabilidad simulada",
  }];
  Object.entries(totals).forEach(([line, item]) => {
    const pick = item.pick || "";
    const probability = pick === "over" ? Number(item.over || 0) : Number(item.under || 0);
    bets.push({
      market: `U/O ${line}`,
      pick: item.label || "-",
      probability,
      agreement: probability / 100,
      detail: "probabilidad simulada",
    });
  });
  const topScore = ((distribution || {}).top_scores || [])[0];
  if (topScore) {
    bets.push({
      market: "Marcador exacto",
      pick: topScore.score || "-",
      probability: Number(topScore.probability || 0),
      agreement: Number(topScore.probability || 0) / 100,
      detail: `N=${formatInteger(distribution.iterations || 0)}`,
    });
  }
  return bets.sort((a, b) => Number(b.probability || 0) - Number(a.probability || 0));
}

function clientScorePanelHtml(distribution, primary) {
  const payload = distribution || {};
  if (!payload.available) return "";
  const topScores = payload.top_scores || [];
  const sourceLabel = primary && primary.mode === "monte_carlo"
    ? `Monte Carlo consenso · N=${formatInteger(payload.iterations || 0)}`
    : "Consenso exacto";
  return `<section class="client-score-panel">
    <header><strong>Marcadores probables</strong><small>${escapeHtml(sourceLabel)}</small></header>
    <div class="top-scores compact">
      ${topScores.slice(0, 3).map((score) => `<span>${escapeHtml(score.score)} <b>${escapeHtml(formatNumber(score.probability ?? 0))}%</b></span>`).join("")}
    </div>
    ${scoreHeatmapDrawerHtml(payload)}
  </section>`;
}

function reportFixtureCardHtml(report) {
  const fixture = report.fixture || {};
  const monteCarlo = report.monte_carlo_consensus || {};
  const consensus = monteCarlo.available ? {
    ...(report.consensus || {}),
    outcome: monteCarlo.outcome || (report.consensus || {}).outcome,
    outcome_label: monteCarlo.outcome_label || (report.consensus || {}).outcome_label,
    outcome_share: Number(monteCarlo.outcome_probability || 0) / 100,
    strength: "Monte Carlo",
  } : report.consensus || {};
  const models = report.models || [];
  const topModels = report.top_models_1x2 || [];
  const stats = report.model_statistics || {};
  const scoreDistribution = (monteCarlo.available ? monteCarlo : report.consensus_score_distribution) || {};
  const calculationLabel = report.sota_calculation_label || (monteCarlo.available ? `SOTA Monte Carlo consenso: N=${formatInteger(monteCarlo.iterations || 0)}` : "Consenso exacto: matriz promedio, sin simulacion");
  const homeAsset = fixture.home_asset || assetFor(fixture.home || "");
  const awayAsset = fixture.away_asset || assetFor(fixture.away || "");
  const consensusClass = ["Baja", ""].includes(consensus.strength || "") ? "low" : "";
  const warnings = report.warnings || [];
  return `<article class="upcoming-card report-fixture-card">
    <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || "")}</strong></header>
    ${fixtureCountdownHtml(fixture)}
    <div class="upcoming-match">
      <div class="upcoming-team">${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
      <span>vs</span>
      <div class="upcoming-team away">${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
    </div>
    <div class="prediction-pick">
      <span>Consenso · ${escapeHtml(consensus.eligible_models || 0)} modelos válidos · ${escapeHtml(calculationLabel)}</span>
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
  const isMonteCarlo = payload.calculation_mode === "monte_carlo";
  const title = isMonteCarlo ? "Monte Carlo sobre matriz consenso" : "Consenso exacto de marcador";
  const subtitle = isMonteCarlo
    ? `N=${formatInteger(payload.iterations || 0)} · ${payload.backend || "numpy"} · λ ${formatNumber(lambdas.home ?? "-")}/${formatNumber(lambdas.away ?? "-")}`
    : `${payload.model_count || 0} modelos · λ ${formatNumber(lambdas.home ?? "-")}/${formatNumber(lambdas.away ?? "-")}`;
  return `<section class="report-panel consensus-score-panel">
    <header>
      <strong>${escapeHtml(title)}</strong>
      <small>${escapeHtml(subtitle)}</small>
    </header>
    <div class="top-scores compact">
      ${topScores.slice(0, 3).map((score) => `<span>${escapeHtml(score.score)} <b>${escapeHtml(formatNumber(score.probability ?? 0))}%</b></span>`).join("")}
    </div>
    ${scoreHeatmapDrawerHtml(payload)}
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
    `${summary.returned || 0}/${summary.requested || 0} partidos - ${summary.group || "Todos"} - Poisson ultimos ${recentLimit}`;
  document.getElementById("upcoming-predictions").innerHTML = (result.predictions || []).map((prediction) => {
    const fixture = prediction.fixture || {};
    const probs = prediction.probabilities || {};
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
    return `<article class="upcoming-card">
      <header><span>${escapeHtml(fixture.date || "")}</span><strong>${escapeHtml(fixture.group || "")}</strong></header>
      <div class="upcoming-match">
        <div class="upcoming-team">${flagHtml(homeAsset)}<strong>${escapeHtml(fixture.home || "")}</strong></div>
        <span>vs</span>
        <div class="upcoming-team away">${flagHtml(awayAsset)}<strong>${escapeHtml(fixture.away || "")}</strong></div>
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
        ${topScores.slice(0, 5).map((score) => `<span>${escapeHtml(score.score)} <b>${escapeHtml(formatNumber(score.probability ?? 0))}%</b></span>`).join("")}
      </div>
    </article>`;
  }).join("") || loadingHtml("Sin fixtures futuros");
  renderTable("upcoming-predictions-table", result.table);
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
  return `<details class="context-poisson context-poisson-drawer">
    <summary>
      <strong>${escapeHtml(title)}</strong>
      <small>${escapeHtml(detail)}</small>
    </summary>
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
    ${scoreHeatmapDrawerHtml(context)}
    ${context.available ? `<details class="recent15-drawer">
      <summary>Ultimos ${escapeHtml(recentLimit)} partidos</summary>
      <div class="recent15-columns">
        ${recentMatchesMiniTable((context.recent_matches || {}).home || [], fixtureData.home || "Local")}
        ${recentMatchesMiniTable((context.recent_matches || {}).away || [], fixtureData.away || "Visitante")}
      </div>
    </details>` : ""}
  </details>`;
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
      const score = cell.score || `${homeGoal}-${awayGoal}`;
      const heat = Math.max(0.04, Math.min(1, Number(cell.probability || 0) / maxProb));
      return `<span title="${escapeAttr(score)}: ${escapeAttr(cell.probability ?? 0)}%" style="--heat:${escapeAttr(heat)}"><em>${escapeHtml(score)}</em><small>${escapeHtml(cell.probability ?? "")}%</small></span>`;
    }).join("")}
  `).join("");
  return `<div class="score-heatmap" style="grid-template-columns: 28px repeat(${awayGoals.length}, minmax(48px, 1fr))">${header}${rows}</div>`;
}

function scoreHeatmapDrawerHtml(contextual) {
  const heatmap = scoreHeatmapHtml(contextual);
  if (!heatmap) return "";
  return `<details class="heatmap-drawer">
    <summary>Heatmap de marcador</summary>
    ${heatmap}
  </details>`;
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
      ...simulationPayload({ iterations, mode: "poisson_live" }),
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
  </article>`;
}

async function runSimulation(mode = "hybrid") {
  clearAlert();
  const poissonLive = mode === "poisson_live";
  document.getElementById("simulation-summary").textContent = poissonLive ? "Ejecutando Monte Carlo Poisson live..." : "Ejecutando Monte Carlo...";
  try {
    const job = await api("/api/mundial/simulate", jsonOptions(simulationPayload({
      mode,
      include_confirmed_results: poissonLive,
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
  const base = kind === "simulation" ? 2000 : 3000;
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
    return;
  }
  const result = job.result || {};
  if (job.kind === "simulation") {
    renderSimulation(result);
  }
  if (job.kind === "upcoming-report") {
    renderUpcomingReport(result);
  }
}

function renderWorldcupJobProgress(kind) {
  const targetId = kind === "simulation"
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
      : [];
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

function formatReportDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("es-MX", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSignedNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value;
  const formatted = formatNumber(Math.abs(number));
  return `${number >= 0 ? "+" : "-"}${formatted}`;
}

function formatInteger(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return value;
  return Math.round(number).toLocaleString("es-MX");
}

function simulationPayload(overrides = {}) {
  return {
    iterations: currentMonteCarloSimulations(),
    seed: Number(document.getElementById("sim-seed").value || 2026),
    poisson_recent_matches: currentPoissonRecentMatches(),
    history_weight: Number(document.getElementById("sim-history-weight").value || 1),
    recency_weight: Number(document.getElementById("sim-recency-weight").value || 0),
    host_advantage: Number(document.getElementById("sim-host-advantage").value || 45),
    max_goals: Number(document.getElementById("sim-max-goals").value || 10),
    score_model: (document.getElementById("sim-score-model") || {}).value || "independent_poisson",
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
  const layers = (summary.poisson_layers || []).join(" + ");
  const recentLimit = config.poisson_recent_matches || currentPoissonRecentMatches();
  const scoreModel = summary.score_model || {};
  document.getElementById("simulation-summary").textContent =
    `${summary.model || "Modelo"} - ${config.iterations || ""} iteraciones - seed ${config.seed || ""} - ${scoreModel.label || config.score_model || "Poisson independiente"} - Poisson ultimos ${recentLimit} - historial ${config.history_weight || ""} - recencia ${config.recency_weight || ""} - ${layers}`;
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
    .replace(/^(CLIError|ValueError|RuntimeError|LineupProviderError):\s*/, "")
    .replace(/\bNone\b/g, "Sin valor");
}
