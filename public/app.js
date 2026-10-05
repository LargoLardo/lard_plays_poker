import { bestHand, blendSparseStrategy, buildPostflopBucket, compareScore, findPreflopStrategy, modelActionMask, modelRaiseTo, PostflopStrategy, v5ActionAmounts, v5ActionMask } from "./model-policy.js";

const RANKS = "23456789TJQKA";
const SUITS = "cdhs";
const SUIT_GLYPH = { c: "♣", d: "♦", h: "♥", s: "♠" };
const STREET_NAMES = ["Preflop", "Flop", "Turn", "River"];
const ACTIONS = ["fold", "check/call", "raise"];
const STORAGE_KEY = "lard-plays-poker-progress-v1";
const DISPLAY_RANKS = [...RANKS].reverse();
const AGENT_DELAY_MS = 2_000;
const NEXT_HAND_DELAY_MS = 8_000;
const SPARSE_NODE_THRESHOLD = 1_500;
const STUDY_MIN_VISITS = 1_000;

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries([
  "modelStatus", "nodesetSelect", "agentSeat", "userSeat", "agentStack", "userStack", "agentPosition", "userPosition", "agentCards", "userCards",
  "agentBet", "userBet", "pot", "board", "message", "street", "toCall", "lastAction", "actions",
  "foldButton", "callButton", "raiseButton", "raiseSlider", "raiseOutput", "finishedActions", "newHandButton", "cancelNextHandButton",
  "newHandTop", "newGameTop", "resultDetail", "nextHandCountdown", "rangePosition", "rangeHistory", "rangeStack", "rangeSize", "rangeGrid", "rangeFold", "rangeCall", "rangeRaise",
  "foldBar", "callBar", "raiseBar", "rangeDetail", "spotCoverage",
  "arenaForm", "arenaA", "arenaB", "arenaHands", "arenaSeed", "arenaRun", "arenaStatus", "arenaResult",
].map((id) => [id, $(id)]));

let model = {};
let postflopStrategy = new PostflopStrategy();
let rangeMode = "all";
let rangeSpots = [];
const nodesets = [{ id:"bundled", label:"Bundled · 10M", preflop:"/preflop-model.json", postflop:"/postflop-model.json" }];
const NODESET_STORAGE_KEY = "lard-plays-poker-nodeset";
let activeNodeset = null;
let modelRequest = null;
const savedProgress = loadProgress();
let handNumber = savedProgress.handNumber;
let bankroll = savedProgress.bankroll;
let game;
let nextHandTimeout = null;
let nextHandTicker = null;
let nextHandDeadline = 0;
let nextHandCancelled = false;
let tableScene = null;

const sceneHost = $("tableScene");
sceneHost.addEventListener("scene-unavailable", () => {
  tableScene = null;
  $("sceneNotice").classList.remove("hidden");
});
import("./table-scene.bundle.js").then(async ({ createTableScene }) => {
  tableScene = await createTableScene($("tableCanvas"), sceneHost);
  if (game) tableScene.update(game);
}).catch((error) => {
  console.warn("The 3D table could not be started:", error);
  $("sceneNotice").classList.remove("hidden");
});

function loadProgress() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (Array.isArray(saved?.bankroll) && saved.bankroll.length === 2 && saved.bankroll.every((value) => Number.isFinite(value) && value >= 0)) {
      return { bankroll: saved.bankroll, handNumber: Number.isInteger(saved.handNumber) ? saved.handNumber : 0 };
    }
  } catch (_) { /* Storage can be unavailable in privacy-restricted browsers. */ }
  return { bankroll: [100, 100], handNumber: 0 };
}

function saveProgress() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ bankroll, handNumber })); } catch (_) { /* Continue without persistence. */ }
}

function deck() {
  const cards = [...RANKS].flatMap((rank) => [...SUITS].map((suit) => rank + suit));
  for (let i = cards.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [cards[i], cards[j]] = [cards[j], cards[i]];
  }
  return cards;
}

function pay(player, amount) {
  const paid = Math.min(amount, game.stacks[player]);
  game.stacks[player] -= paid;
  game.bets[player] += paid;
  game.committed[player] += paid;
  return paid;
}

function newHand() {
  clearNextHandTimer();
  nextHandCancelled = false;
  if (bankroll.some((stack) => stack < 1)) {
    if (game) { game.result = "Game over — start a new game to reset the stacks"; render(); }
    return;
  }
  handNumber++;
  saveProgress();
  const cards = deck();
  const user = handNumber % 2, agent = (handNumber + 1) % 2;
  const stacks = [];
  stacks[user] = bankroll[0];
  stacks[agent] = bankroll[1];
  game = {
    cards, hole: [[cards.pop(), cards.pop()], [cards.pop(), cards.pop()]], board: [],
    stacks, committed: [0, 0], bets: [0, 0], pot: 0,
    user, agent, actor: 1, street: 0,
    pending: new Set([0, 1]), histories: [[], [], [], []], lastRaise: 1,
    publicHistory: [],
    finished: false, reveal: false, lastAction: "Blinds posted", result: "", winner: null,
    nodeset: activeNodeset, preflopModel: model, postflopModel: postflopStrategy,
  };
  pay(0, 1); pay(1, 0.5);
  updateModelStatus();
  logNewHand();
  render();
  if (game.actor === game.agent) scheduleAgent();
}

function logNewHand() {
  const timestamp = new Date().toISOString();
  console.groupCollapsed(`[Lard] Hand #${handNumber} dealt — ${timestamp}`);
  console.log("Timestamp:", timestamp);
  console.log("Lard's cards:", game.hole[game.agent].join(" "));
  console.log("Lard's position:", game.agent === 1 ? "SB" : "BB");
  console.log("Nodeset:", game.nodeset?.label || "Heuristic strategy");
  console.log("Stacks:", { player: `${bankroll[0]} BB`, lard: `${bankroll[1]} BB` });
  console.groupEnd();
}

function startNewGame() {
  const hasProgress = handNumber > 0 || bankroll.some((stack) => stack !== 100);
  if (hasProgress && !window.confirm("Start a new game? This resets both stacks to 100 BB.")) return;
  bankroll = [100, 100];
  handNumber = 0;
  saveProgress();
  newHand();
}

function clearNextHandTimer() {
  if (nextHandTimeout !== null) window.clearTimeout(nextHandTimeout);
  if (nextHandTicker !== null) window.clearInterval(nextHandTicker);
  nextHandTimeout = null;
  nextHandTicker = null;
  nextHandDeadline = 0;
}

function updateNextHandCountdown() {
  if (nextHandDeadline === 0) return;
  const seconds = Math.max(0, Math.ceil((nextHandDeadline - Date.now()) / 1_000));
  ui.nextHandCountdown.textContent = `Next hand in ${seconds} second${seconds === 1 ? "" : "s"}.`;
}

function scheduleNextHand() {
  clearNextHandTimer();
  if (!game?.finished || bankroll.some((stack) => stack < 1)) return;
  const completedGame = game;
  nextHandCancelled = false;
  nextHandDeadline = Date.now() + NEXT_HAND_DELAY_MS;
  updateNextHandCountdown();
  ui.cancelNextHandButton.classList.remove("hidden");
  nextHandTicker = window.setInterval(updateNextHandCountdown, 250);
  nextHandTimeout = window.setTimeout(() => {
    clearNextHandTimer();
    if (game === completedGame && game.finished) newHand();
  }, NEXT_HAND_DELAY_MS);
}

function cancelNextHand() {
  if (nextHandTimeout === null) return;
  clearNextHandTimer();
  nextHandCancelled = true;
  ui.cancelNextHandButton.classList.add("hidden");
  ui.nextHandCountdown.textContent = "Auto-deal cancelled.";
}

function toCall(player = game.actor) { return Math.max(...game.bets) - game.bets[player]; }
function minRaiseTo() { return Math.min(...game.bets.map((bet, player) => bet + game.stacks[player]), Math.max(...game.bets) + game.lastRaise); }
function maxRaiseTo() { return game.bets[game.actor] + game.stacks[game.actor]; }
function fmt(amount) { return `${Number(amount.toFixed(2))} BB`; }

function act(kind, raiseTo = null) {
  if (game.finished || game.actor === null) return;
  const player = game.actor;
  const call = toCall(player);
  if (kind === "fold") {
    if (call <= 0) return;
    game.publicHistory.push([0, 0, 0]);
    game.histories[game.street].push("fold");
    game.lastAction = `${name(player)} folds`;
    finishHand(1 - player, "won by fold");
    return;
  }
  if (kind === "raise") {
    const maximum = maxRaiseTo();
    const target = Math.max(minRaiseTo(), Math.min(Number(raiseTo), maximum));
    const oldHigh = Math.max(...game.bets);
    const potAfterCall = game.pot + game.bets[0] + game.bets[1] + call;
    game.publicHistory.push([2, (target - oldHigh) / potAfterCall, Number(target === maximum)]);
    pay(player, target - game.bets[player]);
    game.lastRaise = Math.max(game.lastRaise, target - oldHigh);
    game.pending = new Set([1 - player]);
    game.histories[game.street].push("raise");
    game.lastAction = `${name(player)} raises to ${fmt(target)}`;
  } else {
    game.publicHistory.push([1, 0, 0]);
    pay(player, call);
    game.pending.delete(player);
    game.histories[game.street].push("check/call");
    game.lastAction = call ? `${name(player)} calls ${fmt(call)}` : `${name(player)} checks`;
  }

  if (game.stacks[1 - player] === 0) game.pending.delete(1 - player);
  if (game.pending.size === 0) closeRound();
  else game.actor = 1 - player;
  render();
  if (!game.finished && game.actor === game.agent) scheduleAgent();
}

function closeRound() {
  // Return an unmatched all-in overage before moving chips into the pot.
  if (game.stacks.some((stack) => stack === 0) && game.bets[0] !== game.bets[1]) {
    const high = game.bets[0] > game.bets[1] ? 0 : 1;
    const excess = Math.abs(game.bets[0] - game.bets[1]);
    game.bets[high] -= excess;
    game.stacks[high] += excess;
  }
  game.pot += game.bets[0] + game.bets[1];
  game.bets = [0, 0];
  if (game.street === 3) return showdown();
  game.street++;
  const dealCount = game.street === 1 ? 3 : 1;
  for (let i = 0; i < dealCount; i++) game.board.push(game.cards.pop());
  if (game.stacks.some((stack) => stack === 0)) {
    while (game.board.length < 5) game.board.push(game.cards.pop());
    game.street = 3;
    return showdown();
  }
  game.actor = 0; // Big blind acts first postflop heads-up.
  game.pending = new Set([0, 1]);
  game.lastRaise = 1;
}

function showdown() {
  const scores = game.hole.map((hole) => bestHand([...hole, ...game.board]));
  game.reveal = true;
  if (compareScore(scores[0], scores[1]) > 0) finishHand(0, scores[0].name, true);
  else if (compareScore(scores[1], scores[0]) > 0) finishHand(1, scores[1].name, true);
  else finishHand(null, `split pot · ${scores[0].name}`, true);
}

function finishHand(winner, reason, potAlreadyCollected = false) {
  if (!potAlreadyCollected) game.pot += game.bets[0] + game.bets[1];
  game.bets = [0, 0];
  if (winner === null) { game.stacks[0] += game.pot / 2; game.stacks[1] += game.pot / 2; }
  else game.stacks[winner] += game.pot;
  bankroll = [game.stacks[game.user], game.stacks[game.agent]];
  saveProgress();
  game.result = winner === null ? "The pot is split" : `${name(winner)} wins ${fmt(game.pot)}`;
  game.lastAction = reason;
  game.finished = true; game.actor = null; game.reveal = true; game.winner = winner;
  render();
  scheduleNextHand();
}

function name(player) { return player === game.user ? "You" : "Lard"; }

function cardMarkup(card, hidden = false) {
  if (hidden) return '<span class="card back" aria-label="Hidden card"></span>';
  const [rank, suit] = card;
  const suitName = { c: "clubs", d: "diamonds", h: "hearts", s: "spades" }[suit];
  const rankName = { T: "10", J: "Jack", Q: "Queen", K: "King", A: "Ace" }[rank] || rank;
  return `<span class="card ${suit === "h" || suit === "d" ? "red" : ""}" aria-label="${rankName} of ${suitName}">${rank === "T" ? "10" : rank}<small>${SUIT_GLYPH[suit]}</small></span>`;
}

function render() {
  const user = game.user, agent = game.agent;
  ui.userStack.textContent = fmt(game.stacks[user]); ui.agentStack.textContent = fmt(game.stacks[agent]);
  ui.userPosition.textContent = user === 1 ? "SB" : "BB"; ui.agentPosition.textContent = agent === 1 ? "SB" : "BB";
  ui.userPosition.classList.toggle("dealer", user === 1); ui.agentPosition.classList.toggle("dealer", agent === 1);
  ui.userSeat.classList.toggle("to-act", !game.finished && game.actor === user);
  ui.agentSeat.classList.toggle("to-act", !game.finished && game.actor === agent);
  ui.userSeat.classList.toggle("hand-winner", game.finished && game.winner === user);
  ui.agentSeat.classList.toggle("hand-winner", game.finished && game.winner === agent);
  ui.userCards.innerHTML = game.hole[user].map((card) => cardMarkup(card)).join("");
  ui.agentCards.innerHTML = game.hole[agent].map((card) => cardMarkup(card, !game.reveal)).join("");
  ui.board.innerHTML = game.board.map((card) => cardMarkup(card)).join("") + Array.from({ length: 5 - game.board.length }, () => '<span class="board-slot"></span>').join("");
  ui.pot.textContent = fmt(game.pot + game.bets[0] + game.bets[1]);
  renderBet(ui.userBet, game.bets[user]); renderBet(ui.agentBet, game.bets[agent]);
  ui.street.textContent = STREET_NAMES[game.street];
  $("handNumber").textContent = `Hand ${String(handNumber).padStart(2, "0")}`;
  ui.lastAction.textContent = game.lastAction;
  const userTurn = !game.finished && game.actor === user;
  const call = userTurn ? toCall(user) : 0;
  ui.toCall.textContent = fmt(call);
  ui.message.textContent = game.finished ? game.result : game.actor === user ? "Your move" : "Lard is thinking…";
  ui.actions.classList.toggle("hidden", game.finished);
  ui.finishedActions.classList.toggle("hidden", !game.finished);
  ui.resultDetail.textContent = game.finished ? game.lastAction : "";
  const gameOver = game.finished && bankroll.some((stack) => stack < 1);
  ui.newHandButton.textContent = gameOver ? "Start a new game" : "Deal now";
  ui.cancelNextHandButton.classList.toggle("hidden", gameOver || nextHandTimeout === null);
  if (gameOver) ui.nextHandCountdown.textContent = "A player is out of chips.";
  else if (game.finished && nextHandCancelled) ui.nextHandCountdown.textContent = "Auto-deal cancelled.";
  for (const button of [ui.foldButton, ui.callButton, ui.raiseButton]) button.disabled = !userTurn;
  ui.foldButton.disabled = !userTurn || call === 0;
  ui.callButton.querySelector("span").textContent = call ? `Call ${fmt(call).replace(" BB", "")}` : "Check";
  ui.raiseSlider.disabled = !userTurn;
  document.querySelectorAll("[data-size]").forEach((button) => { button.disabled = !userTurn; });
  if (userTurn) configureRaise();
  tableScene?.update(game);
}

function renderBet(element, amount) {
  element.classList.toggle("hidden", amount <= 0);
  element.querySelector("strong").textContent = fmt(amount);
}

function configureRaise() {
  const min = minRaiseTo(), max = maxRaiseTo();
  ui.raiseSlider.min = min; ui.raiseSlider.max = max; ui.raiseSlider.step = 0.5;
  if (+ui.raiseSlider.value < min || +ui.raiseSlider.value > max) ui.raiseSlider.value = Math.min(max, Math.max(min, Math.max(...game.bets) + game.pot * 0.5));
  ui.raiseOutput.textContent = fmt(+ui.raiseSlider.value);
  ui.raiseButton.disabled = max <= Math.max(...game.bets) || game.stacks[game.user] <= toCall(game.user);
}

function preflopHand(cards) {
  const ranks = [cards[0][0], cards[1][0]].sort().join("");
  return ranks + (cards[0][1] === cards[1][1] ? "s" : "o");
}

function historyBucket(history) {
  if (!history.length) return "root";
  if (history.length === 1) return history[0] === "check/call" ? "limped" : history[0] === "raise" ? "vs_open" : "root";
  if (history.length === 2) return history.every((a) => a === "raise") ? "vs_3bet" : "vs_open";
  if (history.length === 3) return history.every((a) => a === "raise") ? "vs_4bet" : "vs_3bet";
  return "vs_4bet";
}

function sizeBucket(history) {
  const historyName = historyBucket(history);
  const call = toCall(game.agent);
  if (historyName === "limped") return "Limp";
  if (call > 0 && call <= 2) return "~2.0bb raise";
  if (call <= 2.75) return "~2.75bb raise";
  if (call < 6) return "~6.0bb raise";
  if (call < 10) return "~10.0bb raise";
  if (call < 25) return "~25.0bb raise";
  if (call >= 25) return "Jam (>25.00bb) raise";
  return "Limp";
}

function handStrength(hand) {
  const values = [...hand.slice(0, 2)].map((rank) => RANKS.indexOf(rank) + 2).sort((a, b) => b - a);
  const pair = values[0] === values[1], suited = hand.endsWith("s");
  if ((pair && values[0] >= 11) || (values[0] === 14 && values[1] >= 13) || (suited && values[0] === 14 && values[1] >= 12)) return "premium";
  if (pair || (values[0] === 14 && values[1] >= 10) || (values[0] >= 12 && values[1] >= 10) || (suited && values[0] - values[1] <= 2)) return "playable";
  return "trash";
}

function heuristicPreflop(hand, history, facing) {
  if (!facing) return [0, 0.72, 0.28];
  const strength = handStrength(hand), fourBet = historyBucket(history) === "vs_4bet";
  if (fourBet) return { premium:[0, .25, .75], playable:[.72, .25, .03], trash:[.97, .03, 0] }[strength];
  return { premium:[0, .35, .65], playable:[.28, .55, .17], trash:[.78, .20, .02] }[strength];
}

function preflopDecisionContext() {
  const history = game.histories[0], hand = preflopHand(game.hole[game.agent]);
  const effective = Math.min(...game.stacks);
  const stack = effective < 20 ? "short" : effective < 50 ? "medium" : "deep";
  const bucket = [hand, game.agent === 1 ? "SB" : "BB", stack, historyBucket(history), sizeBucket(history)];
  const key = bucket.join("|");
  const match = findPreflopStrategy(game.preflopModel, bucket, modelActionMask(game));
  const fallback = heuristicPreflop(hand, history, toCall(game.agent) > 0);
  if (!match) return { weights:fallback, spot:key, source:"heuristic fallback (untrained node and no neighboring trained size)" };
  const blended = blendSparseStrategy(match.strategy, fallback, SPARSE_NODE_THRESHOLD);
  const fallbackResolution = [
    match.sizeFallback && `${match.sizeFallback}-size fallback ${match.requestedSize} → ${match.resolvedSize}`,
    match.stackFallback && `${match.stackFallback}-stack fallback ${match.requestedStack} → ${match.resolvedStack}`,
  ].filter(Boolean).join("; ") || "exact requested size and stack";
  return {
    weights:blended.weights,
    spot:match.sizeFallback || match.stackFallback ? `${key}|resolved-node:${match.key}` : key,
    source:blended.heuristicShare
      ? `trained node via ${fallbackResolution} (${match.strategy[3].toLocaleString()} visits; ${(blended.heuristicShare * 100).toFixed(0)}% sparse-node heuristic blend)`
      : `trained node via ${fallbackResolution} (${match.strategy[3].toLocaleString()} visits; no heuristic blend)`,
  };
}

const HISTORY_LABELS = { root:"Unopened — SB acts first", limped:"After SB limp", vs_open:"Facing open", vs_3bet:"Facing 3-bet", vs_4bet:"Facing 4-bet+" };

function rangeHand(row, column) {
  if (row === column) return { label:`${row}${column}`, key:[row, column].sort().join("") + "o", combos:6 };
  const rowIndex = DISPLAY_RANKS.indexOf(row), columnIndex = DISPLAY_RANKS.indexOf(column);
  const suited = columnIndex > rowIndex;
  const high = rowIndex < columnIndex ? row : column;
  const low = high === row ? column : row;
  return { label:`${high}${low}${suited ? "s" : "o"}`, key:[row, column].sort().join("") + (suited ? "s" : "o"), combos:suited ? 4 : 12 };
}

function initializeRangeExplorer() {
  const spots = [...new Set(Object.keys(model).map((key) => key.split("|").slice(1).join("|")))];
  const historyOrder = { root:0, limped:1, vs_open:2, vs_3bet:3, vs_4bet:4 };
  spots.sort((a, b) => {
    const aa = a.split("|"), bb = b.split("|");
    return (historyOrder[aa[2]] ?? 9) - (historyOrder[bb[2]] ?? 9) || aa[0].localeCompare(bb[0]) || a.localeCompare(b);
  });
  rangeSpots = spots.map((spot) => {
    const [position, stack, history, size] = spot.split("|");
    const mask = spot.split("|")[4];
    return { position, stack, history, size, mask };
  });
  populateRangeFilters(0);
  ui.rangeDetail.textContent = activeNodeset?.native
    ? "Hover or tap a hand to see its preflop action mix."
    : "Hover or tap a hand to see its exact action mix.";
  renderRange();
}

function populateRangeFilters(startIndex) {
  const filters = [
    { element:ui.rangePosition, key:"position", label:(value) => value },
    { element:ui.rangeStack, key:"stack", label:(value) => value.replace("_", " ") },
    { element:ui.rangeHistory, key:"history", label:(value) => HISTORY_LABELS[value] || value },
    { element:ui.rangeSize, key:"size", label:(value) => {
      if (ui.rangeHistory.value === "root") return "Blinds posted";
      if (ui.rangeHistory.value === "limped") return "Limp";
      return value;
    } },
  ];
  for (let index = startIndex; index < filters.length; index++) {
    const available = rangeSpots.filter((spot) => filters.slice(0, index).every((filter) => spot[filter.key] === filter.element.value));
    const values = [...new Set(available.map((spot) => spot[filters[index].key]))];
    const current = filters[index].element.value;
    filters[index].element.innerHTML = values.map((value) => `<option value="${value}">${filters[index].label(value)}</option>`).join("");
    if (values.includes(current)) filters[index].element.value = current;
    filters[index].element.disabled = values.length <= 1;
  }
  const sizingCount = ui.rangeSize.options.length;
  ui.spotCoverage.textContent = activeNodeset?.native
    ? "Raise combines small, pot, and all-in choices. Nodes under 1,000 visits are excluded."
    : !sizingCount ? "No averaged preflop nodes yet." : sizingCount === 1
    ? "One sizing was trained for this node; additional sizes require a wider training tree."
    : `${sizingCount} trained sizings are available for this node.`;
}

function renderRange() {
  const spot = [ui.rangePosition.value, ui.rangeStack.value, ui.rangeHistory.value, ui.rangeSize.value].join("|");
  const totals = [0, 0, 0];
  let totalCombos = 0;
  const cells = [];
  const colors = ["#4d90c7", "#4dc78e", "#d75f61"];
  const modeIndex = ACTIONS.indexOf(rangeMode);
  const availableMasks = [...new Set(rangeSpots.filter((s) => [s.position, s.stack, s.history, s.size].join("|") === spot).map((s) => s.mask).filter(Boolean))];

  for (const row of DISPLAY_RANKS) for (const column of DISPLAY_RANKS) {
    const hand = rangeHand(row, column);
    const baseKey = `${hand.key}|${spot}`;
    const rawRows = availableMasks.map((mask) => model[`${baseKey}|${mask}`]).filter(Boolean);
    const rows = rawRows.filter((row) => row[3] >= STUDY_MIN_VISITS);
    // Range display aggregates contexts; actual decisions always match one mask.
    const visits = rows.reduce((sum, row) => sum + row[3], 0);
    const legacy = model[baseKey];
    const node = (legacy?.[3] >= STUDY_MIN_VISITS ? legacy : null) || (visits ? [0, 1, 2].map((a) => rows.reduce((sum, row) => sum + row[a] * row[3], 0) / visits).concat(visits) : null);
    if (!node) {
      const sparse = legacy || rawRows.length;
      const title = sparse ? `${hand.label} — Sparse: fewer than 1,000 visits per node` : `${hand.label} — Untrained`;
      cells.push(`<button class="range-cell missing${sparse ? " sparse" : ""}" disabled title="${title}" aria-label="${title}"><strong>${hand.label}</strong><small>—</small></button>`);
      continue;
    }
    const frequencies = node.slice(0, 3);
    frequencies.forEach((frequency, index) => { totals[index] += frequency * hand.combos; });
    totalCombos += hand.combos;
    let background;
    let shownFrequency;
    if (rangeMode === "all") {
      const foldEnd = frequencies[0] * 100, callEnd = (frequencies[0] + frequencies[1]) * 100;
      background = `linear-gradient(90deg,${colors[0]} 0 ${foldEnd}%,${colors[1]} ${foldEnd}% ${callEnd}%,${colors[2]} ${callEnd}% 100%)`;
      shownFrequency = Math.max(...frequencies);
    } else {
      shownFrequency = frequencies[modeIndex];
      const alpha = .08 + shownFrequency * .92;
      background = `color-mix(in srgb, ${colors[modeIndex]} ${alpha * 100}%, #101c18)`;
    }
    const title = `${hand.label} — Fold ${(frequencies[0] * 100).toFixed(1)}%, Call ${(frequencies[1] * 100).toFixed(1)}%, Raise ${(frequencies[2] * 100).toFixed(1)}% · ${node[3].toLocaleString()} visits`;
    cells.push(`<button class="range-cell" style="background:${background}" title="${title}" aria-label="${title}"><strong>${hand.label}</strong><small>${Math.round(shownFrequency * 100)}%</small></button>`);
  }
  ui.rangeGrid.innerHTML = cells.join("");
  ui.rangeGrid.querySelectorAll(".range-cell:not(.missing)").forEach((cell) => {
    cell.addEventListener("click", () => { ui.rangeDetail.textContent = cell.getAttribute("aria-label"); });
    cell.addEventListener("mouseenter", () => { ui.rangeDetail.textContent = cell.getAttribute("aria-label"); });
  });
  const averages = totals.map((total) => totalCombos ? total / totalCombos : 0);
  [ui.rangeFold, ui.rangeCall, ui.rangeRaise].forEach((element, index) => { element.textContent = `${(averages[index] * 100).toFixed(1)}%`; });
  [ui.foldBar, ui.callBar, ui.raiseBar].forEach((element, index) => { element.style.width = `${averages[index] * 100}%`; });
}

function postflopDecisionContext() {
  const features = buildPostflopBucket(game, 100);
  const equity = features.equity;
  const facing = toCall(game.agent) > 0;
  const potOdds = facing ? toCall(game.agent) / (game.pot + game.bets[0] + game.bets[1] + toCall(game.agent)) : 0;
  let heuristic;
  if (!facing) heuristic = [0, Math.max(.32, .82 - equity * .55), Math.min(.68, .18 + equity * .55)];
  else if (equity > .72) heuristic = [0, .38, .62];
  else if (equity + .06 >= potOdds) heuristic = [.12, .76, .12];
  else heuristic = [.78, .21, .01];

  const match = game.postflopModel.find(features.bucket, modelActionMask(game));
  const diagnostic = `${STREET_NAMES[game.street]}|bucket:${features.key}|equity:${(equity * 100).toFixed(1)}%|ppot:${(features.ppot * 100).toFixed(1)}%|npot:${(features.npot * 100).toFixed(1)}%`;
  if (!match) return { weights:heuristic, spot:diagnostic, source:"heuristic fallback (no trained node in this context)" };

  const blended = blendSparseStrategy(match.strategy, heuristic);
  const fallbackResolution = [
    match.sizeFallback && `${match.sizeFallback}-size fallback ${match.requestedSize} → ${match.resolvedSize}`,
    match.stackFallback && `${match.stackFallback}-stack fallback ${match.requestedStack} → ${match.resolvedStack}`,
  ].filter(Boolean);
  const handResolution = match.distance === 0 ? "trained postflop node" : `nearest trained postflop hand bucket (distance ${match.distance})`;
  const matchType = match.exact ? `exact ${handResolution}` : `${handResolution}${fallbackResolution.length ? `; ${fallbackResolution.join("; ")}` : ""}`;
  return {
    weights:blended.weights,
    spot:diagnostic,
    source:blended.heuristicShare
      ? `${matchType}; ${match.strategy[3].toLocaleString()} visits; ${(blended.heuristicShare * 100).toFixed(0)}% sparse-node heuristic blend`
      : `${matchType}; ${match.strategy[3].toLocaleString()} visits; no heuristic blend`,
  };
}

function normalizeWeights(weights) {
  const total = weights.reduce((sum, weight) => sum + Math.max(0, weight), 0);
  if (total <= 0) return [0, 1, 0];
  return weights.map((weight) => Math.max(0, weight) / total);
}

function logAgentDecision(context, frequencies, decision) {
  const timestamp = new Date().toISOString();
  console.group(`[Lard decision] ${timestamp}`);
  console.log("Timestamp:", timestamp);
  console.log("Cards:", game.hole[game.agent].join(" "));
  console.log("Spot:", context.spot);
  console.log("Strategy source:", context.source);
  console.log(
    `Action frequencies: Fold ${(frequencies[0] * 100).toFixed(2)}% | `
    + `Call ${(frequencies[1] * 100).toFixed(2)}% | Raise ${(frequencies.slice(2).reduce((sum, value) => sum + value, 0) * 100).toFixed(2)}%`,
  );
  console.table({
    Fold: { frequency:`${(frequencies[0] * 100).toFixed(2)}%` },
    Call: { frequency:`${(frequencies[1] * 100).toFixed(2)}%` },
    Raise: { frequency:`${(frequencies.slice(2).reduce((sum, value) => sum + value, 0) * 100).toFixed(2)}%` },
  });
  console.log("Ultimate decision:", decision);
  console.groupEnd();
}

function scheduleAgent() {
  render();
  const scheduledGame = game;
  window.setTimeout(async () => {
    if (game !== scheduledGame || game.finished || game.actor !== game.agent) return;
    if (game.nodeset?.native) {
      let report;
      try {
        const response = await fetch("/api/decision", {
          method:"POST", headers:{ "Content-Type":"application/json" },
          body:JSON.stringify({ nodeset:game.nodeset.id, hero:game.hole[game.agent], board:game.board, history:game.publicHistory, actor:game.agent, street:game.street }),
        });
        report = await response.json();
        if (!response.ok) throw new Error(report.error || "Native inference failed");
      } catch (error) {
        console.warn("V5 native inference unavailable:", error);
        report = { weights:[0, 1, 0, 0, 0], trained:false, unavailable:true };
      }
      if (game !== scheduledGame || game.finished || game.actor !== game.agent) return;
      const mask = v5ActionMask(game);
      const frequencies = normalizeWeights(report.weights.map((weight, action) => mask & (1 << action) ? weight : 0));
      const choice = weightedChoice(frequencies);
      const target = v5ActionAmounts(game)[choice];
      logAgentDecision({ spot:`${STREET_NAMES[game.street]}|history:${report.history || "off-tree"}|cluster:${report.bucket ?? "none"}`,
        source:report.trained ? `V5 averaged strategy; ${report.visits} visits${report.translated ? "; translated opposing size" : ""}` : report.unavailable ? "V5 off-tree check/call fallback" : "V5 uniform legal fallback" }, frequencies,
        choice >= 2 ? `Raise to ${fmt(target)}` : choice === 1 ? "Check/call" : "Fold");
      if (choice >= 2) act("raise", target);
      else act(ACTIONS[choice]);
      return;
    }
    const context = game.street === 0 ? preflopDecisionContext() : postflopDecisionContext();
    const weights = [...context.weights];
    const mask = modelActionMask(game);
    if (!(mask & 1)) weights[0] = 0;
    if (!(mask & 4)) weights[2] = 0;
    const frequencies = normalizeWeights(weights);
    const choice = weightedChoice(frequencies);
    if (choice === 2) {
      const target = modelRaiseTo(game);
      logAgentDecision(context, frequencies, `Raise to ${fmt(Math.min(target, maxRaiseTo()))}`);
      act("raise", Math.min(target, maxRaiseTo()));
    } else {
      const decision = choice === 1 ? (toCall(game.agent) > 0 ? `Call ${fmt(toCall(game.agent))}` : "Check") : "Fold";
      logAgentDecision(context, frequencies, decision);
      act(ACTIONS[choice]);
    }
  }, AGENT_DELAY_MS);
}

function weightedChoice(weights) {
  const total = weights.reduce((a, b) => a + Math.max(0, b), 0);
  let value = Math.random() * total;
  for (let i = 0; i < weights.length; i++) if ((value -= Math.max(0, weights[i])) <= 0) return i;
  return 1;
}

ui.foldButton.addEventListener("click", () => act("fold"));
ui.callButton.addEventListener("click", () => act("check/call"));
ui.raiseButton.addEventListener("click", () => act("raise", +ui.raiseSlider.value));
ui.raiseSlider.addEventListener("input", () => { ui.raiseOutput.textContent = fmt(+ui.raiseSlider.value); });
document.querySelectorAll("[data-size]").forEach((button) => button.addEventListener("click", () => {
  const size = button.dataset.size;
  const target = size === "allin" ? maxRaiseTo() : Math.max(...game.bets) + (game.pot + game.bets[0] + game.bets[1]) * Number(size);
  ui.raiseSlider.value = Math.min(maxRaiseTo(), Math.max(minRaiseTo(), target));
  ui.raiseOutput.textContent = fmt(+ui.raiseSlider.value);
}));
ui.newHandButton.addEventListener("click", () => bankroll.some((stack) => stack < 1) ? startNewGame() : newHand());
ui.cancelNextHandButton.addEventListener("click", cancelNextHand);
ui.newHandTop.addEventListener("click", newHand);
ui.newGameTop.addEventListener("click", startNewGame);
ui.nodesetSelect.addEventListener("change", () => loadNodeset(ui.nodesetSelect.value));
ui.arenaForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (ui.arenaRun.disabled) return;
  const selected = [ui.arenaA, ui.arenaB].map((select) => ({ id:select.value, label:select.selectedOptions[0].textContent }));
  ui.arenaRun.disabled = true;
  ui.arenaResult.replaceChildren();
  ui.arenaStatus.textContent = "Running duplicate deals…";
  try {
    const response = await fetch("/api/arena", {
      method:"POST", headers:{ "Content-Type":"application/json" },
      body:JSON.stringify({ a:selected[0].id, b:selected[1].id, hands:Number(ui.arenaHands.value), seed:Number(ui.arenaSeed.value) }),
    });
    const report = await response.json();
    if (!response.ok) throw new Error(report.error || "Match failed");
    ui.arenaResult.innerHTML = '<table><thead><tr><th>Checkpoint</th><th>Net BB</th><th>BB/100 · 95% interval</th><th>Wins / losses / ties</th><th>Trained decisions</th></tr></thead><tbody></tbody></table>';
    [report.a, report.b].forEach((agent, index) => {
      const row = document.createElement("tr");
      const interval = agent.ci95_bb_per_100?.map((number) => number.toFixed(2)).join(" to ") || "not enough pairs";
      const c = agent.coverage;
      for (const value of [selected[index].label, agent.net_bb.toFixed(2), `${agent.bb_per_100.toFixed(2)} (${interval})`,
        `${agent.wins.toLocaleString()} / ${agent.losses.toLocaleString()} / ${agent.ties.toLocaleString()}`,
        `${c.trained.toLocaleString()} / ${c.decisions.toLocaleString()}${c.translated_actions ? ` · ${c.translated_actions.toLocaleString()} translated` : ""}${c.off_tree_decisions ? ` · ${c.off_tree_decisions.toLocaleString()} off-tree` : ""}`]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      ui.arenaResult.querySelector("tbody").append(row);
    });
    const interval = report.a.ci95_bb_per_100;
    const conclusion = !interval ? "Run more pairs for an interval." : interval[0] > 0 ? "A leads in this match." : interval[1] < 0 ? "B leads in this match." : "No clear winner at 95% confidence.";
    ui.arenaStatus.textContent = `${report.hands.toLocaleString()} hands · ${report.duplicate_pairs.toLocaleString()} duplicate deals · ${report.samples} equity samples · ${report.elapsed_seconds.toFixed(2)}s. ${conclusion}`;
  } catch (error) {
    ui.arenaStatus.textContent = error.message;
  } finally {
    ui.arenaRun.disabled = false;
  }
});
[ui.rangePosition, ui.rangeStack, ui.rangeHistory, ui.rangeSize].forEach((filter, index) => filter.addEventListener("change", () => {
  populateRangeFilters(index + 1);
  renderRange();
}));
document.querySelectorAll("[data-range-action]").forEach((button) => button.addEventListener("click", () => {
  rangeMode = button.dataset.rangeAction;
  document.querySelectorAll("[data-range-action]").forEach((item) => item.classList.toggle("active", item === button));
  renderRange();
}));
document.querySelectorAll(".session-popover button").forEach((button) => button.addEventListener("click", () => { $("sessionMenu").open = false; }));
document.addEventListener("click", (event) => {
  if (!$("sessionMenu").contains(event.target)) $("sessionMenu").open = false;
});
function activateTab(button) {
  document.querySelectorAll("[data-tab]").forEach((tab) => {
    const selected = tab === button;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== button.dataset.tab));
  history.replaceState(null, "", `#${button.dataset.tab}`);
}
document.querySelectorAll("[data-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button)));
document.querySelector(".view-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const tabs = [...document.querySelectorAll("[data-tab]")];
  const index = tabs.indexOf(document.activeElement);
  const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
  activateTab(tabs[next]);
  tabs[next].focus();
});
const linkedTab = document.querySelector(`[data-tab="${location.hash.slice(1)}"]`);
if (linkedTab) activateTab(linkedTab);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("sessionMenu").open = false;
  if (event.target.matches("input, select, textarea") || $("gameplayPanel").classList.contains("hidden") || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key.toLowerCase() === "f" && !ui.foldButton.disabled) ui.foldButton.click();
  if (event.key.toLowerCase() === "c" && !ui.callButton.disabled) ui.callButton.click();
  if (event.key.toLowerCase() === "r" && !ui.raiseButton.disabled) ui.raiseButton.click();
});

function updateModelStatus() {
  if (modelRequest) return;
  // Counts come from the loaded export; the current hand retains its own policy.
  ui.modelStatus.lastChild.textContent = activeNodeset
    ? ` ${activeNodeset.label} · ${activeNodeset.nodeCount.toLocaleString()} nodes${game && game.nodeset?.id !== activeNodeset.id ? " · applies next hand" : ""}`
    : " Heuristic strategy";
}

async function loadNodeset(id) {
  const selected = nodesets.find((item) => item.id === id);
  if (!selected) return;
  modelRequest?.abort();
  const request = new AbortController();
  modelRequest = request;
  ui.nodesetSelect.setAttribute("aria-busy", "true");
  ui.modelStatus.lastChild.textContent = ` Loading ${selected.label}…`;
  try {
    const [preflopNodes, postflopNodes] = await Promise.all([selected.preflop, selected.postflop].map(async (path) => {
      const response = await fetch(path, { signal:request.signal });
      if (!response.ok) throw new Error("Strategy unavailable");
      return response.json();
    }));
    const nextPostflop = new PostflopStrategy(postflopNodes);
    if (modelRequest !== request) return;
    model = preflopNodes;
    postflopStrategy = nextPostflop;
    activeNodeset = { ...selected, nodeCount:selected.totalNodes ?? Object.keys(preflopNodes).length + Object.keys(postflopNodes).length };
    ui.nodesetSelect.value = id;
    ui.modelStatus.classList.add("ready");
    initializeRangeExplorer();
    try { localStorage.setItem(NODESET_STORAGE_KEY, id); } catch (_) { /* Selection still works without storage. */ }
  } catch (error) {
    if (modelRequest !== request) return;
    ui.nodesetSelect.value = activeNodeset?.id || "bundled";
    ui.modelStatus.lastChild.textContent = ` Could not load ${selected.label}.${activeNodeset ? ` Still using ${activeNodeset.label}.` : " Using heuristics."}`;
    if (!activeNodeset) ui.rangeGrid.innerHTML = '<p class="range-error">The trained strategy could not be loaded.</p>';
    console.warn("Nodeset could not be loaded:", error);
  } finally {
    if (modelRequest === request) {
      modelRequest = null;
      ui.nodesetSelect.removeAttribute("aria-busy");
      if (activeNodeset?.id === id) updateModelStatus();
    }
  }
}

try {
  const response = await fetch("/api/nodesets", { cache:"no-store" });
  if (response.ok) {
    nodesets.push(...await response.json());
    ui.arenaRun.disabled = false;
    ui.arenaStatus.textContent = "Choose two checkpoints to compare.";
  }
} catch (_) { /* Static hosting still offers the bundled model. */ }
ui.nodesetSelect.replaceChildren(...nodesets.map((item) => new Option(item.label, item.id)));
const arenaChoices = [...nodesets, { id:"baseline:random", label:"Random legal bet sizes" }, { id:"baseline:call", label:"Always check/call" }, { id:"baseline:pot", label:"Random actions with pot bets" }];
for (const select of [ui.arenaA, ui.arenaB]) select.replaceChildren(...arenaChoices.map((item) => new Option(item.label, item.id)));
const latest = nodesets[1];
if (latest) {
  ui.arenaA.value = latest.id;
  ui.arenaB.value = nodesets.find((item) => item.id !== latest.id && item.iterations && item.iterations !== latest.iterations)?.id || "bundled";
}
ui.nodesetSelect.disabled = nodesets.length < 2;
let selectedId = "bundled";
try { selectedId = localStorage.getItem(NODESET_STORAGE_KEY) || selectedId; } catch (_) { /* Use the bundled default. */ }
await loadNodeset(nodesets.some((item) => item.id === selectedId) ? selectedId : "bundled");
if (!activeNodeset && selectedId !== "bundled") await loadNodeset("bundled");
newHand();
