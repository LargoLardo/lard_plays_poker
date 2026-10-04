import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  bestHand,
  blendSparseStrategy,
  buildPostflopBucket,
  findPreflopStrategy,
  modelActionMask,
  modelRaiseTo,
  PostflopStrategy,
  postflopStreetFromBucket,
} from "../public/model-policy.js";

assert.equal(bestHand(["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"]).name, "straight flush");

const certainQuads = {
  street:1,
  agent:1,
  hole:[["Kh", "Kd"], ["As", "Ad"]],
  board:["Ah", "Ac", "2d"],
  bets:[0, 0],
  stacks:[100, 100],
  pot:10,
  histories:[[], [], [], []],
};
const features = buildPostflopBucket(certainQuads, 10, () => .5);
assert.deepEqual(features.bucket, [[7, 0, 0, "rainbow", true], "SB", 0, "small", "mid_deep"]);

const mature = blendSparseStrategy([.8, .1, .1, 100], [0, 1, 0]);
assert.deepEqual(mature, { weights:[.8, .1, .1], heuristicShare:0 });
const sparse = blendSparseStrategy([.8, .1, .1, 50], [0, 1, 0]);
assert.deepEqual(sparse, { weights:[.4, .55, .05], heuristicShare:.5 });

const preflopHigher = [0, 0, 1, 500];
const preflopLower = [0, 1, 0, 500];
const preflopNodes = {
  "AKo|BB|deep|vs_open|~10.0bb raise":preflopHigher,
  "AKo|BB|deep|vs_open|~2.0bb raise":preflopLower,
};
const preflopFallback = findPreflopStrategy(preflopNodes, ["AKo", "BB", "deep", "vs_open", "~6.0bb raise"]);
assert.equal(preflopFallback.sizeFallback, "higher");
assert.equal(preflopFallback.resolvedSize, "~10.0bb raise");
assert.deepEqual(preflopFallback.strategy, preflopHigher);
delete preflopNodes["AKo|BB|deep|vs_open|~10.0bb raise"];
assert.equal(findPreflopStrategy(preflopNodes, ["AKo", "BB", "deep", "vs_open", "~6.0bb raise"]).resolvedSize, "~2.0bb raise");

const preflopStackNodes = {
  "AKo|BB|deep|vs_open|~6.0bb raise":preflopHigher,
  "AKo|BB|short|vs_open|~6.0bb raise":preflopLower,
  "AKo|BB|medium|vs_open|~10.0bb raise":[1, 0, 0, 500],
};
const preflopStackFallback = findPreflopStrategy(preflopStackNodes, ["AKo", "BB", "medium", "vs_open", "~6.0bb raise"]);
assert.equal(preflopStackFallback.stackFallback, "deeper");
assert.equal(preflopStackFallback.resolvedStack, "deep");
assert.equal(preflopStackFallback.resolvedSize, "~6.0bb raise", "exact size has priority over a higher-size exact-stack node");
delete preflopStackNodes["AKo|BB|deep|vs_open|~6.0bb raise"];
assert.equal(findPreflopStrategy(preflopStackNodes, ["AKo", "BB", "medium", "vs_open", "~6.0bb raise"]).stackFallback, "shallower");

const sampleHandBucket = [4, 1, 0, "rainbow", false];
const lowerPostflopKey = JSON.stringify([sampleHandBucket, "BB", 0, "small", "deep"]);
const higherPostflopKey = JSON.stringify([sampleHandBucket, "BB", 0, "large", "deep"]);
const postflopSizeStrategy = new PostflopStrategy({ [lowerPostflopKey]:preflopLower, [higherPostflopKey]:preflopHigher });
const postflopFallback = postflopSizeStrategy.find([sampleHandBucket, "BB", 0, "medium", "deep"]);
assert.equal(postflopFallback.sizeFallback, "higher");
assert.equal(postflopFallback.resolvedSize, "large");
assert.deepEqual(postflopFallback.strategy, preflopHigher);

const postflopStackNodes = {
  [JSON.stringify([sampleHandBucket, "BB", 0, "medium", "mid_deep"])]:preflopHigher,
  [JSON.stringify([sampleHandBucket, "BB", 0, "medium", "deep"])]:[1, 0, 0, 500],
  [JSON.stringify([sampleHandBucket, "BB", 0, "medium", "short"])]:preflopLower,
  [JSON.stringify([sampleHandBucket, "BB", 0, "large", "mid"])]:[.5, .5, 0, 500],
};
const postflopStackStrategy = new PostflopStrategy(postflopStackNodes);
const postflopStackFallback = postflopStackStrategy.find([sampleHandBucket, "BB", 0, "medium", "mid"]);
assert.equal(postflopStackFallback.stackFallback, "deeper");
assert.equal(postflopStackFallback.resolvedStack, "mid_deep");
assert.equal(postflopStackFallback.resolvedSize, "medium", "exact size has priority over a higher-size exact-stack node");
delete postflopStackNodes[JSON.stringify([sampleHandBucket, "BB", 0, "medium", "mid_deep"])];
delete postflopStackNodes[JSON.stringify([sampleHandBucket, "BB", 0, "medium", "deep"])];
const postflopShallowerStrategy = new PostflopStrategy(postflopStackNodes);
assert.equal(postflopShallowerStrategy.find([sampleHandBucket, "BB", 0, "medium", "mid"]).stackFallback, "shallower");

// Exact, nearest-hand, size, and stack fallback must preserve action availability.
const preflopBucket = ["AKo", "BB", "deep", "vs_open", "~6.0bb raise"];
const maskedPreflop = { [preflopBucket.join("|") + "|7"]:preflopHigher };
assert.equal(findPreflopStrategy(maskedPreflop, preflopBucket, 3), null);
maskedPreflop[preflopBucket.join("|") + "|3"] = preflopLower;
assert.deepEqual(findPreflopStrategy(maskedPreflop, preflopBucket, 3).strategy, preflopLower);
assert.deepEqual(findPreflopStrategy(maskedPreflop, preflopBucket, 7).strategy, preflopHigher);
for (const size of ["small", "large"]) for (const stack of ["mid", "deep"]) {
  const bucket = [sampleHandBucket, "BB", 0, size, stack];
  const masked = new PostflopStrategy({ [JSON.stringify([...bucket, 7])]:preflopHigher });
  assert.equal(masked.find([[5, 1, 0, "rainbow", false], "BB", 0, "medium", "mid_deep"], 3), null);
}
const sameBucket = [sampleHandBucket, "BB", 0, "medium", "deep"];
const bothMasks = new PostflopStrategy({
  [JSON.stringify([...sameBucket, 3])]:preflopLower,
  [JSON.stringify([...sameBucket, 7])]:preflopHigher,
});
assert.deepEqual(bothMasks.find(sameBucket, 3).strategy, preflopLower);
assert.deepEqual(bothMasks.find(sameBucket, 7).strategy, preflopHigher);
assert.deepEqual(bothMasks.find([[5, ...sampleHandBucket.slice(1)], ...sameBucket.slice(1)], 3).strategy, preflopLower);

const riverBase = { street:3, agent:1, histories:[[], [], [], ["raise", "raise", "raise"]], lastRaise:20 };
assert.equal(modelActionMask({ ...riverBase, stacks:[0, 55], bets:[93, 38], pot:14 }), 3);
assert.equal(modelActionMask({ ...riverBase, stacks:[55, 75], bets:[38, 18], pot:14 }), 7);
const flopBase = { street:1, agent:1, histories:[[], [], [], []], stacks:[100, 100], bets:[0, 0], lastRaise:1 };
assert.equal(modelRaiseTo({ ...flopBase, pot:5 }), 2, "half pot rounds 2.5 to even");
assert.equal(modelRaiseTo({ ...flopBase, pot:7 }), 4, "half pot rounds 3.5 to even");
assert.equal(modelActionMask({ ...flopBase, pot:5 }), 6, "free check cannot fold");

const nodes = JSON.parse(await readFile("public/postflop-model.json", "utf8"));
const strategy = new PostflopStrategy(nodes);
const streetCounts = { flop:0, turn:0, river:0 };
for (const key of Object.keys(nodes)) streetCounts[postflopStreetFromBucket(JSON.parse(key))]++;
assert.ok(Object.values(streetCounts).every((count) => count > 1_000));

for (const street of Object.keys(streetCounts)) {
  const key = Object.keys(nodes).find((candidate) => postflopStreetFromBucket(JSON.parse(candidate)) === street);
  const match = strategy.find(JSON.parse(key));
  assert.equal(match.exact, true);
  assert.deepEqual(match.strategy, nodes[key]);
}

console.log(`Model policy tests passed (${JSON.stringify(streetCounts)}).`);
