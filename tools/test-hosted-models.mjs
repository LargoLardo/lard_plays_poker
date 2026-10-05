import assert from "node:assert/strict";
import { createServer } from "node:http";
import { readFile, mkdtemp, rm, stat } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { resolve, join } from "node:path";
import { tmpdir } from "node:os";
import { chromium } from "playwright";
import handler, { inference } from "../api/decision.js";
import { decisionLine } from "./hosted-inference.mjs";

const directory = await mkdtemp(join(tmpdir(), "lard-hosted-test-"));
inference.temporary = directory;
const base = { nodeset:"v3", hero:["As", "Kd"], board:[], history:[], actor:1, street:0 };
const requests = [base, { ...base, hero:["2c", "2d"] }, { ...base, history:[[1,0,0]], actor:0 }];
for (let street = 1; street <= 3; street++) requests.push({ ...base, actor:0, street,
  board:["2s", "7d", "Tc", "Jh", "3c"].slice(0, street + 2), history:Array.from({ length:street * 2 }, () => [1,0,0]) });
// Compare every street to inference directly from the original, full checkpoint.
const nativeName = `train_v5${process.platform === "win32" ? ".exe" : ""}`;
const expected = execFileSync(`cpp/build/${nativeName}`, ["--resume", "checkpoints/v3.bin", "--infer", "--memory-mb", "1700", "--cache-mb", "8"],
  { input:requests.map(decisionLine).join("\n") + "\n", encoding:"utf8", timeout:60_000 }).trim().split("\n").map(JSON.parse);
const publicDirectory = resolve("public");
let decisions = 0;
const server = createServer(async (request, response) => {
  const pathname = new URL(request.url, "http://localhost").pathname;
  if (pathname === "/api/decision") {
    let body = "";
    for await (const chunk of request) body += chunk;
    request.body = body;
    response.status = (code) => { response.statusCode = code; return response; };
    response.json = (value) => { response.setHeader("Content-Type", "application/json"); response.end(JSON.stringify(value)); };
    decisions++;
    await handler(request, response);
    return;
  }
  const file = resolve(publicDirectory, "." + (pathname === "/" ? "/index.html" : pathname));
  try {
    assert.ok(file.startsWith(publicDirectory + "/"));
    const contents = await readFile(file);
    response.setHeader("Content-Type", file.endsWith(".js") ? "text/javascript" : file.endsWith(".html") ? "text/html" : file.endsWith(".css") ? "text/css" : file.endsWith(".json") ? "application/json" : "application/octet-stream");
    response.end(contents);
  } catch (_) { response.statusCode = 404; response.end(); }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const url = `http://127.0.0.1:${server.address().port}`;
const post = (body) => fetch(url + "/api/decision", { method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(body) });
let browser;
try {
  const coldStarted = performance.now();
  const actual = await Promise.all(requests.map(async (body) => {
    const response = await post(body);
    assert.equal(response.status, 200);
    return response.json();
  }));
  assert.deepEqual(actual, expected, "Hosted inference matches the original checkpoint on every street, including concurrent requests");
  assert.ok(expected.every((report) => report.trained));
  const engine = inference.engine;
  assert.deepEqual(await (await post(base)).json(), expected[0]);
  assert.equal(inference.engine, engine, "Reuse the warm C++ model");
  assert.equal(await (await fetch(url + "/api/decision")).json().then((x) => x.error), "Use POST");
  for (const invalid of [{ ...base, hero:["As", "As"] }, { ...base, nodeset:"v2" }, { ...base, street:1 },
    { ...base, history:[[2,"nan",0]] }, { ...base, actor:2 }]) assert.equal((await post(invalid)).status, 400);
  const seconds = ((performance.now() - coldStarted) / 1000).toFixed(2);
  const packageBytes = (await stat("artifacts/hosted/v3.bin.gz")).size + (await stat(`artifacts/hosted/${nativeName}`)).size;
  assert.ok(packageBytes < 240 * 1024 * 1024);
  console.log(`Hosted native parity passed; cold start plus concurrent decisions ${seconds}s; native assets ${(packageBytes / 1048576).toFixed(1)} MiB.`);
  browser = await chromium.launch({ args:["--enable-webgl", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
  const page = await browser.newPage({ viewport:{ width:1440, height:960 } });
  await page.addInitScript(() => localStorage.setItem("lard-plays-poker-nodeset", "v3"));
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(url);
  await page.waitForSelector("#modelStatus.ready", { state:"attached" });
  assert.deepEqual(await page.locator("#nodesetSelect option").allTextContents(), ["V1 · 10M", "V2 · 500M", "V3 · 968.272M"]);
  assert.equal(await page.locator("#arenaRun").isDisabled(), true, "Hosted catalog does not pretend to provide local arena execution");
  const before = decisions;
  await page.locator("#callButton").click();
  await page.waitForFunction(() => !document.getElementById("callButton").disabled || !document.getElementById("finishedActions").classList.contains("hidden"));
  assert.ok(decisions > before, "V3 Play calls the hosted native endpoint");
  await page.locator("#studyTab").click();
  await page.locator("#advancedRaises").check();
  assert.equal(await page.locator("#raiseBreakdown").isVisible(), true);
  await page.locator("#sessionMenu summary").click();
  await page.selectOption("#nodesetSelect", "v2");
  await page.waitForFunction(() => !document.getElementById("nodesetSelect").hasAttribute("aria-busy"));
  assert.equal(await page.locator("#advancedRaises").isDisabled(), true);
  assert.match(await page.locator("#modelStatus").textContent(), /V2 · 500M/);
  await page.selectOption("#nodesetSelect", "bundled");
  await page.waitForFunction(() => !document.getElementById("nodesetSelect").hasAttribute("aria-busy"));
  assert.match(await page.locator("#modelStatus").textContent(), /V1 · 10M/);
  await page.selectOption("#nodesetSelect", "v3");
  await page.waitForFunction(() => !document.getElementById("nodesetSelect").hasAttribute("aria-busy"));
  await page.locator("#sessionMenu summary").click();
  await page.locator("#advancedRaises").check();
  await page.screenshot({ path:"artifacts/hosted-models-study.png", fullPage:true });
  assert.deepEqual(errors, []);
  console.log("Hosted browser checks passed: catalog fallback, V1/V2/V3 selection, V3 Play, and advanced raise Study.");
} finally {
  await browser?.close();
  inference.close();
  await new Promise((done) => server.close(done));
  await rm(directory, { recursive:true, force:true });
}
