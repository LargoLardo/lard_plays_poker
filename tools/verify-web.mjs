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
const catalog = JSON.parse(await readFile("public/nodesets.json", "utf8"));
if (catalog.map((item) => item.id).join(",") !== "v2,v3") throw new Error("Hosted model catalog must contain V2 and V3");
for (const item of catalog) {
  const preflop = JSON.parse(await readFile(`public${item.preflop}`, "utf8"));
  const postflop = JSON.parse(await readFile(`public${item.postflop}`, "utf8"));
  if (!Object.keys(preflop).length) throw new Error(`${item.label} has no Study data`);
  if (item.native) {
    if (!Object.values(preflop).every((row) => row.length === 7 && Math.abs(row[2] - row[4] - row[5] - row[6]) < 1e-9)) {
      throw new Error("V3 Study must retain all three raise sizes");
    }
  } else if (!Object.keys(postflop).length) throw new Error("V2 has no postflop policy");
}
await Promise.all([`artifacts/hosted/train_v5${process.platform === "win32" ? ".exe" : ""}`, "artifacts/hosted/v3.bin.gz"].map((file) => access(file)));
console.log("V1, V2, and V3 are available in hosted builds.");
