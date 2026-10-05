const RANKS = "23456789TJQKA";
const SUITS = "cdhs";
const PREFLOP_SIZES = ["Limp", "~2.0bb raise", "~2.75bb raise", "~6.0bb raise", "~10.0bb raise", "~25.0bb raise", "Jam (>25.00bb) raise"];
const PREFLOP_STACKS = ["short", "medium", "deep"];
const POSTFLOP_SIZES = ["small", "medium", "large", "overbet"];
const POSTFLOP_STACKS = ["short", "mid", "mid_deep", "deep"];

function prioritizedLevels(levels, requested, higherLabel, lowerLabel) {
  const requestedIndex = levels.indexOf(requested);
  if (requestedIndex < 0) return [{ value:requested, fallback:null }];
  return [
    { value:requested, fallback:null },
    ...levels.slice(requestedIndex + 1).map((value) => ({ value, fallback:higherLabel })),
    ...levels.slice(0, requestedIndex).reverse().map((value) => ({ value, fallback:lowerLabel })),
  ];
}

export function findPreflopStrategy(nodes, bucket, mask = 7) {
  const sizes = prioritizedLevels(PREFLOP_SIZES, bucket[4], "higher", "lower");
  const stacks = prioritizedLevels(PREFLOP_STACKS, bucket[2], "deeper", "shallower");
  for (const size of sizes) {
    for (const stack of stacks) {
      const candidate = [...bucket];
      candidate[2] = stack.value;
      candidate[4] = size.value;
      const baseKey = candidate.join("|");
      const key = nodes[`${baseKey}|${mask}`] ? `${baseKey}|${mask}` : baseKey;
      if (nodes[key]) return {
        strategy:nodes[key],
        key,
        exact:!size.fallback && !stack.fallback,
        requestedSize:bucket[4],
        resolvedSize:size.value,
        sizeFallback:size.fallback,
        requestedStack:bucket[2],
        resolvedStack:stack.value,
        stackFallback:stack.fallback,
      };
    }
  }
  return null;
}

export function blendSparseStrategy(strategy, heuristic, threshold = 100) {
  const visits = strategy[3];
  if (visits >= threshold) return { weights:strategy.slice(0, 3), heuristicShare:0 };
  const modelConfidence = Math.max(0, visits) / threshold;
  return {
    weights:strategy.slice(0, 3).map((weight, index) => modelConfidence * weight + (1 - modelConfidence) * heuristic[index]),
    heuristicShare:1 - modelConfidence,
  };
}

export function compareScore(a, b) {
  if (a.rank !== b.rank) return a.rank - b.rank;
  for (let index = 0; index < Math.max(a.kickers.length, b.kickers.length); index++) {
    if ((a.kickers[index] || 0) !== (b.kickers[index] || 0)) return (a.kickers[index] || 0) - (b.kickers[index] || 0);
  }
  return 0;
}

function scoreFive(cards) {
  const values = cards.map((card) => RANKS.indexOf(card[0]) + 2).sort((a, b) => b - a);
  const counts = new Map();
  values.forEach((value) => counts.set(value, (counts.get(value) || 0) + 1));
  const groups = [...counts].sort((a, b) => b[1] - a[1] || b[0] - a[0]);
  const flush = cards.every((card) => card[1] === cards[0][1]);
  const unique = [...new Set(values)];
  if (unique[0] === 14) unique.push(1);
  let straight = 0;
  for (let index = 0; index <= unique.length - 5; index++) {
    if (unique[index] - unique[index + 4] === 4) straight = Math.max(straight, unique[index]);
  }
  if (flush && straight) return { rank:8, kickers:[straight], name:"straight flush" };
  if (groups[0][1] === 4) return { rank:7, kickers:[groups[0][0], groups[1][0]], name:"four of a kind" };
  if (groups[0][1] === 3 && groups[1][1] === 2) return { rank:6, kickers:[groups[0][0], groups[1][0]], name:"full house" };
  if (flush) return { rank:5, kickers:values, name:"flush" };
  if (straight) return { rank:4, kickers:[straight], name:"straight" };
  if (groups[0][1] === 3) return { rank:3, kickers:[groups[0][0], ...groups.slice(1).map((group) => group[0]).sort((a, b) => b - a)], name:"three of a kind" };
  if (groups[0][1] === 2 && groups[1][1] === 2) return { rank:2, kickers:[Math.max(groups[0][0], groups[1][0]), Math.min(groups[0][0], groups[1][0]), groups[2][0]], name:"two pair" };
  if (groups[0][1] === 2) return { rank:1, kickers:[groups[0][0], ...groups.slice(1).map((group) => group[0]).sort((a, b) => b - a)], name:"pair" };
  return { rank:0, kickers:values, name:"high card" };
}

export function bestHand(cards) {
  let best = null;
  for (let a = 0; a < cards.length - 4; a++) for (let b = a + 1; b < cards.length - 3; b++)
    for (let c = b + 1; c < cards.length - 2; c++) for (let d = c + 1; d < cards.length - 1; d++)
      for (let e = d + 1; e < cards.length; e++) {
        const score = scoreFive([cards[a], cards[b], cards[c], cards[d], cards[e]]);
        if (!best || compareScore(score, best) > 0) best = score;
      }
  return best;
}

function sampleCards(available, count, rng) {
  const cards = [...available];
  for (let index = 0; index < count; index++) {
    const selected = index + Math.floor(rng() * (cards.length - index));
    [cards[index], cards[selected]] = [cards[selected], cards[index]];
  }
  return cards.slice(0, count);
}

export function estimatePostflopFeatures(hero, board, samples = 100, rng = Math.random) {
  const known = new Set([...hero, ...board]);
  const available = [...RANKS].flatMap((rank) => [...SUITS].map((suit) => rank + suit)).filter((card) => !known.has(card));
  const cardsToDeal = 5 - board.length;
  let wins = 0, ties = 0;
  let aheadNowBehindLater = 0, behindNowAheadLater = 0, aheadNowTotal = 0, behindNowTotal = 0;

  for (let iteration = 0; iteration < samples; iteration++) {
    const sample = sampleCards(available, 2 + cardsToDeal, rng);
    const villain = sample.slice(0, 2);
    const runout = [...board, ...sample.slice(2)];
    const finalComparison = compareScore(bestHand([...hero, ...runout]), bestHand([...villain, ...runout]));
    if (finalComparison > 0) wins++;
    else if (finalComparison === 0) ties++;

    const currentComparison = compareScore(bestHand([...hero, ...board]), bestHand([...villain, ...board]));
    if (currentComparison > 0) {
      aheadNowTotal++;
      if (finalComparison < 0) aheadNowBehindLater++;
    } else if (currentComparison < 0) {
      behindNowTotal++;
      if (finalComparison > 0) behindNowAheadLater++;
    }
  }
  return {
    equity:(wins + .5 * ties) / samples,
    ppot:behindNowTotal ? behindNowAheadLater / behindNowTotal : 0,
    npot:aheadNowTotal ? aheadNowBehindLater / aheadNowTotal : 0,
  };
}

function flushTexture(board) {
  const counts = new Map();
  board.forEach((card) => counts.set(card[1], (counts.get(card[1]) || 0) + 1));
  const maximum = Math.max(...counts.values());
  if (maximum >= 3) return "monotone";
  if (maximum === 2) return "two_tone";
  return "rainbow";
}

function isPaired(board) {
  return new Set(board.map((card) => card[0])).size !== board.length;
}

function maximumSuited(board) {
  if (!board.length) return 0;
  const counts = new Map();
  board.forEach((card) => counts.set(card[1], (counts.get(card[1]) || 0) + 1));
  return Math.max(...counts.values());
}

function flushDrawCompleted(previous, current) {
  return maximumSuited(current) >= 3 && maximumSuited(previous) < 3;
}

function maximumStraightDraw(board) {
  if (!board.length) return 0;
  const indices = new Set(board.map((card) => RANKS.indexOf(card[0])));
  let best = 0;
  for (let low = 0; low < 9; low++) {
    let hits = 0;
    for (let rank = low; rank < low + 5; rank++) if (indices.has(rank)) hits++;
    best = Math.max(best, hits);
  }
  if (indices.has(12)) {
    let wheel = 1;
    for (let rank = 0; rank < 4; rank++) if (indices.has(rank)) wheel++;
    best = Math.max(best, wheel);
  }
  return best;
}

function straightDrawCompleted(previous, current) {
  return maximumStraightDraw(current) >= 3 && maximumStraightDraw(previous) < 3;
}

function historyBucket(history) {
  if (!history.length) return "root";
  if (history.length === 1) return history[0] === "check/call" ? "limped" : history[0] === "raise" ? "vs_open" : "root";
  if (history.length === 2) return history.every((action) => action === "raise") ? "vs_3bet" : "vs_open";
  if (history.length === 3) return history.every((action) => action === "raise") ? "vs_4bet" : "vs_3bet";
  return "vs_4bet";
}

function roundEven(value) {
  const floor = Math.floor(value);
  return value - floor === .5 ? floor + floor % 2 : Math.round(value);
}

export function modelRaiseTo(game, player = game.agent) {
  const high = Math.max(...game.bets);
  const maximum = game.stacks[player] + game.bets[player];
  const history = game.histories[game.street];
  if (game.street !== 1 && historyBucket(history) === "vs_4bet") return maximum;
  const totalPot = game.pot + game.bets[0] + game.bets[1];
  return Math.min(maximum, roundEven(game.street === 0 ? high * 3 : high + totalPot * .5));
}

export function modelActionMask(game, player = game.agent) {
  const high = Math.max(...game.bets);
  const maximum = game.stacks[player] + game.bets[player];
  const effective = Math.min(maximum, game.stacks[1 - player] + game.bets[1 - player]);
  const minimum = Math.min(effective, high + game.lastRaise);
  const amount = modelRaiseTo(game, player);
  const canRaise = game.stacks[1 - player] > 0 && maximum > high && amount > high && amount >= minimum && amount <= maximum;
  return 2 | (game.bets[player] < high ? 1 : 0) | (canRaise ? 4 : 0);
}

// V5 inference runs natively; these are the same five action sizes applied to
// the actual browser stacks, including legal/duplicate-size filtering.
export function v5ActionAmounts(game, player = game.agent) {
  const high = Math.max(...game.bets);
  const maximum = game.stacks[player] + game.bets[player];
  const pot = game.pot + game.bets[0] + game.bets[1] + high - game.bets[player];
  const raises = game.histories[game.street].filter((action) => action === "raise").length;
  const rounded = (value) => Math.min(maximum, roundEven(value * 2) / 2);
  return [0, 0, rounded(game.street === 0 ? high * (raises ? 3 : 2.5) : high + pot * .5), rounded(high + pot), maximum];
}

export function v5ActionMask(game, player = game.agent) {
  const high = Math.max(...game.bets);
  const maximum = game.stacks[player] + game.bets[player];
  const effective = Math.min(maximum, game.stacks[1 - player] + game.bets[1 - player]);
  const minimum = Math.min(effective, high + game.lastRaise);
  const raises = game.histories[game.street].filter((action) => action === "raise").length;
  const amounts = v5ActionAmounts(game, player);
  let mask = 2 | (game.bets[player] < high ? 1 : 0);
  for (let action = 2; action < 5; action++) {
    if (action !== 4 && raises >= 2) continue;
    if (game.stacks[1 - player] <= 0 || maximum <= high || amounts[action] <= high || amounts[action] < minimum) continue;
    if (amounts.slice(2, action).some((amount, index) => (mask & (1 << (index + 2))) && amount === amounts[action])) continue;
    mask |= 1 << action;
  }
  return mask;
}

function sizeBucket(toCall, pot) {
  const ratio = toCall / pot;
  if (ratio < .4) return "small";
  if (ratio < .75) return "medium";
  if (ratio < 1.1) return "large";
  return "overbet";
}

function sprBucket(stack, pot) {
  const spr = stack / pot;
  if (spr > 10) return "deep";
  if (spr > 4) return "mid_deep";
  if (spr > 1.5) return "mid";
  return "short";
}

function previousStreetRaise(agent, history) {
  const parity = agent === 0 ? 0 : 1;
  return history.some((action, index) => index % 2 === parity && action === "raise");
}

export function postflopStreetFromBucket(bucket) {
  return { 5:"flop", 7:"turn", 4:"river" }[bucket[0].length];
}

export function buildPostflopBucket(game, samples = 100, rng = Math.random) {
  const { street, agent, board, bets, stacks, pot, histories } = game;
  const features = estimatePostflopFeatures(game.hole[agent], board, samples, rng);
  const equityBucket = Math.min(Math.floor(features.equity * 8), 7);
  const potentialBucket = (value) => Math.min(Math.floor(value * 4), 3);
  const position = agent === 0 ? "BB" : "SB";
  const totalPot = pot + bets[0] + bets[1];
  const call = Math.max(...bets) - Math.min(...bets);
  const size = sizeBucket(call, totalPot);
  const spr = sprBucket(stacks[agent], totalPot);
  let handBucket, bucket;

  if (street === 1) {
    handBucket = [equityBucket, potentialBucket(features.ppot), potentialBucket(features.npot), flushTexture(board), isPaired(board)];
    bucket = [handBucket, position, Math.min(histories[1].filter((action) => action === "raise").length, 3), size, spr];
  } else if (street === 2) {
    handBucket = [equityBucket, potentialBucket(features.ppot), potentialBucket(features.npot), flushTexture(board), isPaired(board), flushDrawCompleted(board.slice(0, 3), board), straightDrawCompleted(board.slice(0, 3), board)];
    bucket = [handBucket, position, historyBucket(histories[2]), size, spr, previousStreetRaise(agent, histories[1])];
  } else if (street === 3) {
    handBucket = [equityBucket, flushDrawCompleted(board.slice(0, 4), board), straightDrawCompleted(board.slice(0, 4), board), isPaired(board)];
    bucket = [handBucket, position, historyBucket(histories[3]), size, spr, previousStreetRaise(agent, histories[2])];
  } else {
    throw new Error(`Cannot build a postflop bucket for street ${street}`);
  }
  return { ...features, bucket, key:JSON.stringify(bucket), street:postflopStreetFromBucket(bucket) };
}

function handBucketDistance(left, right) {
  let distance = Math.abs(left[0] - right[0]) * 3;
  for (let index = 1; index < Math.max(left.length, right.length); index++) {
    if (typeof left[index] === "number" && typeof right[index] === "number") distance += Math.abs(left[index] - right[index]);
    else if (left[index] !== right[index]) distance += 1;
  }
  return distance;
}

export class PostflopStrategy {
  constructor(nodes = {}) {
    this.nodes = nodes;
    this.contexts = new Map();
    for (const [key, strategy] of Object.entries(nodes)) {
      const bucket = JSON.parse(key);
      const contextKey = `${postflopStreetFromBucket(bucket)}|${JSON.stringify(bucket.slice(1))}`;
      if (!this.contexts.has(contextKey)) this.contexts.set(contextKey, []);
      this.contexts.get(contextKey).push({ bucket, strategy, key });
    }
  }

  find(bucket, mask = 7) {
    const sizes = prioritizedLevels(POSTFLOP_SIZES, bucket[3], "higher", "lower");
    const stacks = prioritizedLevels(POSTFLOP_STACKS, bucket[4], "deeper", "shallower");
    for (const size of sizes) {
      for (const stack of stacks) {
        const candidateBucket = [...bucket];
        candidateBucket[3] = size.value;
        candidateBucket[4] = stack.value;
        const baseKey = JSON.stringify(candidateBucket);
        const maskedKey = JSON.stringify([...candidateBucket, mask]);
        const key = this.nodes[maskedKey] ? maskedKey : baseKey;
        const resolution = {
          sizeFallback:size.fallback,
          requestedSize:bucket[3],
          resolvedSize:size.value,
          stackFallback:stack.fallback,
          requestedStack:bucket[4],
          resolvedStack:stack.value,
        };
        if (this.nodes[key]) return { strategy:this.nodes[key], key, exact:!size.fallback && !stack.fallback, distance:0, ...resolution };
        const match = this.findNearestInContext(candidateBucket, mask);
        if (match) return { ...match, ...resolution };
      }
    }
    return null;
  }

  findNearestInContext(bucket, mask = 7) {
    const contextKey = `${postflopStreetFromBucket(bucket)}|${JSON.stringify(bucket.slice(1))}`;
    const maskedContext = `${postflopStreetFromBucket(bucket)}|${JSON.stringify([...bucket.slice(1), mask])}`;
    const candidates = this.contexts.get(maskedContext) || this.contexts.get(contextKey) || [];
    let nearest = null;
    for (const candidate of candidates) {
      const distance = handBucketDistance(bucket[0], candidate.bucket[0]);
      if (!nearest || distance < nearest.distance || (distance === nearest.distance && candidate.strategy[3] > nearest.strategy[3])) {
        nearest = { ...candidate, exact:false, distance };
      }
    }
    return nearest;
  }
}
