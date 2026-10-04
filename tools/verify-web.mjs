import { access, readFile } from "node:fs/promises";

const required = ["public/index.html", "public/app.js", "public/model-policy.js", "public/styles.css", "public/assets/hands/left.glb", "public/assets/hands/right.glb", "public/assets/hands/LICENSE.txt", "public/table-scene.bundle.js", "public/preflop-model.json", "public/postflop-model.json"];
await Promise.all(required.map((file) => access(file)));
const model = JSON.parse(await readFile("public/preflop-model.json", "utf8"));
if (Object.keys(model).length < 1000) throw new Error("Preflop model export looks incomplete");
const spots = new Map();
for (const key of Object.keys(model)) {
  const spot = key.split("|").slice(1).join("|");
  spots.set(spot, (spots.get(spot) || 0) + 1);
}
if ([...spots.values()].some((count) => count !== 169)) throw new Error("A range-explorer spot is missing starting hands");
const postflopModel = JSON.parse(await readFile("public/postflop-model.json", "utf8"));
if (Object.keys(postflopModel).length < 45_000) throw new Error("Postflop model export looks incomplete");
const streets = new Set(Object.keys(postflopModel).map((key) => ({ 5:"flop", 7:"turn", 4:"river" })[JSON.parse(key)[0].length]));
if (!["flop", "turn", "river"].every((street) => streets.has(street))) throw new Error("Postflop export does not cover every street");
console.log(`Static app ready (${(Object.keys(model).length + Object.keys(postflopModel).length).toLocaleString()} full-game nodes across all four streets).`);
