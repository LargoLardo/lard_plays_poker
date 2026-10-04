import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { resolve, extname } from "node:path";
import { chromium } from "playwright";

const root = resolve("public");
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = createServer(async (req, res) => {
  const pathname = new URL(req.url, "http://localhost").pathname;
  const path = resolve(root, `.${pathname === "/" ? "/index.html" : pathname}`);
  try {
    if (!path.startsWith(`${root}/`)) throw new Error("Invalid path");
    const contents = await readFile(path);
    res.writeHead(200, { "Content-Type": types[extname(path)] || "application/octet-stream" });
    res.end(contents);
  } catch {
    res.writeHead(404); res.end();
  }
});
const browser = await chromium.launch({ args: ["--enable-webgl", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const url = `http://127.0.0.1:${server.address().port}`;
await mkdir("artifacts", { recursive: true });
const errors = [];

async function pageFor(options = {}) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, ...options });
  page.on("pageerror", (error) => errors.push(error.message));
  return page;
}
async function waitForMove(page) {
  await page.waitForFunction(() => !document.getElementById("callButton").disabled);
}
async function assertFits(page) {
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "The page must not scroll horizontally");
  for (const id of ["agentSeat", "userSeat", "potLabel"]) {
    const box = await page.locator(`#${id}`).boundingBox();
    assert.ok(box.x >= 0 && box.x + box.width <= page.viewportSize().width, `${id} must stay inside the viewport`);
  }
}
async function screenshot(page, name) {
  await page.screenshot({ path: `artifacts/${name}.png`, fullPage: true });
}
async function forceCalls(page) {
  for (const file of ["preflop-model.json", "postflop-model.json"]) {
    const nodes = JSON.parse(await readFile(`public/${file}`, "utf8"));
    for (const key of Object.keys(nodes)) nodes[key] = [0, 1, 0, 50_000];
    await page.route(`**/${file}`, (route) => route.fulfill({ json: nodes }));
  }
}

try {
  // Load the real model first, to catch import and initialization regressions.
  const smoke = await pageFor();
  const handAssets = new Set();
  smoke.on("response", (response) => {
    if (response.url().endsWith(".glb") && response.ok()) handAssets.add(response.url().split("/").at(-1));
  });
  await smoke.goto(url);
  await smoke.waitForSelector(".scene-ready");
  await smoke.waitForFunction(() => document.querySelectorAll("#userCards .card").length === 2);
  assert.equal(await smoke.locator("#tableScene").getAttribute("data-view"), "first-person");
  assert.equal(await smoke.locator("#gameplayPanel h1, #gameplayPanel h2").count(), 0);
  assert.ok((await smoke.locator("#userCards").boundingBox()).width <= 1, "The held cards replace the flat card overlay");
  assert.equal(await smoke.locator("#agentCards .back").count(), 2);
  assert.equal(await smoke.locator("#modelStatus.ready").count(), 1);
  assert.deepEqual([...handAssets].sort(), ["left.glb", "right.glb"], "Both local hand meshes load");
  assert.equal(await smoke.locator(".hand-meta #potLabel").count(), 1, "Pot stays in the controls");
  await smoke.waitForTimeout(700);
  await assertFits(smoke);
  await screenshot(smoke, "table-desktop");
  await smoke.close();

  // Switching updates Study now and Play next hand, and a failed load is atomic.
  const switching = await pageFor();
  await switching.clock.install({ time: new Date("2026-10-04T12:00:00Z") });
  await switching.clock.pauseAt(new Date("2026-10-04T12:00:01Z"));
  await switching.addInitScript(() => { Math.random = () => .5; });
  await forceCalls(switching);
  const raisedPreflop = JSON.parse(await readFile("public/preflop-model.json", "utf8"));
  const raisedPostflop = JSON.parse(await readFile("public/postflop-model.json", "utf8"));
  for (const nodes of [raisedPreflop, raisedPostflop]) for (const key of Object.keys(nodes)) nodes[key] = [0, 0, 1, 50_000];
  await switching.route("**/api/nodesets", (route) => route.fulfill({ json:[
    { id:"raising", label:"Test raising", preflop:"/raising-pre.json", postflop:"/raising-post.json" },
    { id:"sparse", label:"Test sparse", preflop:"/sparse-pre.json", postflop:"/raising-post.json" },
    { id:"broken", label:"Unavailable", preflop:"/raising-pre.json", postflop:"/missing-post.json" },
  ] }));
  await switching.route("**/raising-pre.json", (route) => route.fulfill({ json:raisedPreflop }));
  await switching.route("**/raising-post.json", (route) => route.fulfill({ json:raisedPostflop }));
  await switching.route("**/missing-post.json", (route) => route.fulfill({ status:500 }));
  await switching.route("**/sparse-pre.json", (route) => route.fulfill({ json:{
    "AKo|SB|deep|root|~2.0bb raise|7":[0, 0, 1, 999],
    "AAo|SB|deep|root|~2.0bb raise|7":[0, 0, 1, 1000],
    "KKo|SB|deep|root|~2.0bb raise|3":[1, 0, 0, 600],
    "KKo|SB|deep|root|~2.0bb raise|7":[0, 0, 1, 600],
    "QQo|SB|deep|root|~2.0bb raise|3":[1, 0, 0, 999],
    "QQo|SB|deep|root|~2.0bb raise|7":[0, 0, 1, 1000],
    "22o|SB|deep|root|~2.0bb raise":[0, 1, 0, 999],
    "33o|SB|deep|root|~2.0bb raise":[0, 1, 0, 1000],
  } }));
  let arenaRequest;
  await switching.route("**/api/arena", (route) => {
    arenaRequest = JSON.parse(route.request().postData());
    const agent = { net_bb:0, bb_per_100:0, ci95_bb_per_100:[-1, 1], wins:10, losses:10, ties:0, coverage:{ decisions:100, trained:90 } };
    return route.fulfill({ json:{ hands:20, duplicate_pairs:10, samples:3, elapsed_seconds:.01, a:agent, b:agent } });
  });
  await switching.goto(url);
  await switching.waitForSelector("#modelStatus.ready", { state:"attached" });
  await switching.locator("#sessionMenu summary").click();
  await switching.selectOption("#nodesetSelect", "raising");
  await switching.waitForFunction(() => !document.getElementById("nodesetSelect").hasAttribute("aria-busy"));
  assert.match(await switching.locator("#modelStatus").textContent(), /applies next hand/);
  await switching.locator("#studyTab").click();
  assert.equal(await switching.locator("#rangeRaise").textContent(), "100.0%");
  await switching.locator("#sessionMenu summary").click();
  await switching.selectOption("#nodesetSelect", "broken");
  await switching.waitForFunction(() => !document.getElementById("nodesetSelect").hasAttribute("aria-busy"));
  assert.equal(await switching.locator("#nodesetSelect").inputValue(), "raising");
  assert.equal(await switching.locator("#rangeRaise").textContent(), "100.0%");
  assert.match(await switching.locator("#modelStatus").textContent(), /Still using Test raising/);
  await switching.selectOption("#nodesetSelect", "sparse");
  await switching.waitForFunction(() => !document.getElementById("nodesetSelect").hasAttribute("aria-busy"));
  for (const hand of ["AKo", "KK", "22"]) {
    const cell = switching.locator(`#rangeGrid [aria-label^="${hand} "]`);
    assert.equal(await cell.isDisabled(), true, `${hand} is below 1,000 visits per node`);
    assert.match(await cell.getAttribute("class"), /sparse/);
    assert.equal(await cell.locator("small").textContent(), "—");
  }
  for (const hand of ["AA", "QQ", "33"]) {
    assert.equal(await switching.locator(`#rangeGrid [aria-label^="${hand} "]`).isDisabled(), false, "Exactly 1,000 visits is visible");
  }
  assert.match(await switching.locator('#rangeGrid [aria-label^="QQ "]').getAttribute("aria-label"), /Raise 100.0%/, "Sparse masks are excluded before aggregation");
  await switching.selectOption("#nodesetSelect", "raising");
  await switching.waitForFunction(() => !document.getElementById("nodesetSelect").hasAttribute("aria-busy"));
  await switching.locator("#playTab").click();
  await switching.locator("#callButton").click();
  await switching.clock.runFor(4100);
  assert.equal(await switching.locator("#street").textContent(), "Flop", "The existing hand keeps its call policy");
  assert.match(await switching.locator("#lastAction").textContent(), /checks/, "Postflop also keeps the previous model");
  await switching.locator("#sessionMenu summary").click();
  await switching.locator("#newHandTop").click();
  await switching.clock.runFor(2100);
  assert.match(await switching.locator("#lastAction").textContent(), /raises/, "The next hand uses the selected model");
  await switching.reload();
  await switching.waitForSelector("#modelStatus.ready", { state:"attached" });
  assert.equal(await switching.locator("#nodesetSelect").inputValue(), "raising", "Selection survives reload");
  await switching.setViewportSize({ width:320, height:568 });
  await switching.locator("#sessionMenu summary").click();
  const picker = await switching.locator("#nodesetSelect").boundingBox();
  assert.ok(picker.x >= 0 && picker.x + picker.width <= 320, "The selector fits on a phone");
  await switching.locator("#arenaTab").click();
  await switching.selectOption("#arenaA", "raising");
  await switching.selectOption("#arenaB", "sparse");
  await switching.locator("#arenaHands").fill("20");
  await switching.locator("#arenaSeed").fill("9");
  await switching.locator("#arenaRun").click();
  await switching.waitForSelector("#arenaResult tbody tr");
  assert.deepEqual(arenaRequest, { a:"raising", b:"sparse", hands:20, seed:9 });
  assert.equal(await switching.locator("#arenaResult tbody tr").count(), 2);
  assert.match(await switching.locator("#arenaStatus").textContent(), /No clear winner/);
  assert.equal(await switching.locator("#arenaRun").isDisabled(), false);
  assert.ok(await switching.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Arena fits on a phone");
  await switching.close();

  // Force the trained action mix to check/call so every street is repeatable.
  const page = await pageFor();
  await page.addInitScript(() => { Math.random = () => .5; });
  await forceCalls(page);
  await page.goto(url);
  await page.waitForSelector(".scene-ready");
  await waitForMove(page);
  assert.equal(await page.locator("#pot").textContent(), "1.5 BB");
  await page.locator("#callButton").click();
  assert.equal(await page.locator("#raiseSlider").isDisabled(), true, "Sizing must pause on the opponent's turn");
  assert.equal(await page.locator("#tableScene").getAttribute("data-action"), "call");
  await page.waitForSelector('#tableScene[data-animating="true"]');
  await page.waitForTimeout(140);
  await screenshot(page, "table-call-motion");
  await waitForMove(page);
  let playingTableHeight;
  for (const [street, count] of [["Flop", 3], ["Turn", 4], ["River", 5]]) {
    assert.equal(await page.locator("#street").textContent(), street);
    assert.equal(await page.locator("#board .card").count(), count);
    assert.equal(await page.locator("#tableScene").getAttribute("data-board-count"), String(count));
    assert.equal(await page.locator("#agentCards .back").count(), 2);
    if (street === "Flop") {
      await page.waitForTimeout(700);
      await screenshot(page, "table-flop");
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForTimeout(300);
      await assertFits(page);
      await screenshot(page, "table-mobile");
      await page.setViewportSize({ width: 320, height: 568 });
      await page.waitForTimeout(300);
      await assertFits(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollHeight <= innerHeight), "Phone actions fit without scrolling");
      await screenshot(page, "table-small-phone");
      await page.setViewportSize({ width: 1440, height: 960 });
    }
    if (street === "River") {
      await page.setViewportSize({ width: 320, height: 568 });
      await page.waitForTimeout(700);
      await assertFits(page);
      await screenshot(page, "table-phone-river");
      await page.setViewportSize({ width: 1440, height: 960 });
    }
    playingTableHeight = (await page.locator("#tableScene").boundingBox()).height;
    await page.locator("#callButton").click();
    if (street !== "River") await waitForMove(page);
  }
  await page.waitForSelector("#finishedActions:not(.hidden)");
  assert.equal((await page.locator("#tableScene").boundingBox()).height, playingTableHeight, "Showdown must not resize the table mid-reveal");
  assert.equal(await page.locator("#agentCards .back").count(), 0);
  assert.equal(await page.locator("#tableScene").getAttribute("data-revealed"), "true");
  const stacks = await page.locator("#userStack, #agentStack").allTextContents();
  assert.equal(stacks.reduce((sum, text) => sum + parseFloat(text), 0), 200, "Showdown conserves chips");
  await page.locator("#cancelNextHandButton").click();
  assert.equal(await page.locator("#nextHandCountdown").textContent(), "Auto-deal cancelled.");
  await page.waitForSelector('#tableScene[data-animating="false"]');
  await screenshot(page, "table-showdown");

  // A new hand replaces the scene and flips the dealer; raising moves chips.
  await page.locator("#newHandButton").click();
  await waitForMove(page);
  assert.equal(await page.locator("#handNumber").textContent(), "Hand 02");
  assert.equal(await page.locator("#userPosition").textContent(), "BB");
  assert.equal(await page.locator("#tableScene").getAttribute("data-board-count"), "0");
  await page.locator('[data-size="1"]').click();
  const target = await page.locator("#raiseOutput").textContent();
  await page.locator("#raiseButton").click();
  assert.equal(await page.locator("#userBet strong").textContent(), target);
  assert.equal(await page.locator("#tableScene").getAttribute("data-action"), "raise");
  await page.waitForSelector('#tableScene[data-animating="true"]');
  await page.waitForTimeout(140);
  await screenshot(page, "table-raise-motion");
  await waitForMove(page);

  // Study still loads its full matrix and supports keyboard tab navigation.
  await page.locator("#studyTab").click();
  assert.equal(await page.locator("#rangeGrid .range-cell").count(), 169);
  await page.locator('[data-range-action="raise"]').click();
  await page.locator("#rangeGrid .range-cell:not(.missing)").first().click();
  assert.match(await page.locator("#rangeDetail").textContent(), /visits/);
  await screenshot(page, "table-study");
  await page.locator("#studyTab").focus();
  await page.keyboard.press("ArrowLeft");
  assert.equal(await page.locator("#playTab").getAttribute("aria-selected"), "true");

  // Test a fold with real controls and stop the automatic next deal.
  await page.locator("#sessionMenu summary").click();
  await page.locator("#newHandTop").click();
  await waitForMove(page);
  await page.keyboard.press("f");
  await page.waitForSelector("#finishedActions:not(.hidden)");
  assert.equal(await page.locator("#lastAction").textContent(), "won by fold");
  assert.equal(await page.locator("#tableScene").getAttribute("data-action"), "fold");
  await page.locator("#cancelNextHandButton").click();
  // Interrupt the fold/payout; the next hand must clear every old motion.
  await page.locator("#newHandButton").click();
  await waitForMove(page);
  await page.waitForSelector('#tableScene[data-animating="false"]');
  assert.equal(await page.locator("#tableScene").getAttribute("data-board-count"), "0");
  assert.equal(await page.locator("#tableScene").getAttribute("data-revealed"), "false");
  await screenshot(page, "table-after-interrupted-fold");
  // Full-stack transfers and the automatic runout must conserve the bankroll.
  await page.setViewportSize({ width: 320, height: 568 });
  const phoneTableHeight = (await page.locator("#tableScene").boundingBox()).height;
  await page.locator('[data-size="allin"]').click();
  await page.locator("#raiseButton").click();
  await page.waitForSelector("#finishedActions:not(.hidden)");
  await page.locator("#cancelNextHandButton").click();
  await page.waitForSelector('#tableScene[data-animating="false"]');
  assert.equal((await page.locator("#tableScene").boundingBox()).height, phoneTableHeight, "Phone controls must keep the scene stable through showdown");
  const allInStacks = await page.locator("#userStack, #agentStack").allTextContents();
  assert.equal(allInStacks.reduce((sum, text) => sum + parseFloat(text), 0), 200);
  assert.equal(await page.locator("#tableScene").getAttribute("data-board-count"), "5");
  await screenshot(page, "table-all-in");
  await page.close();

  // A quick check must not cut a deal short; a new hand must cancel an old reveal.
  const motion = await pageFor();
  await motion.clock.install({ time: new Date("2026-10-03T12:00:00Z") });
  await motion.clock.pauseAt(new Date("2026-10-03T12:00:01Z"));
  await motion.addInitScript(() => {
    Math.random = () => .5;
    localStorage.setItem("lard-plays-poker-progress-v1", JSON.stringify({ handNumber: 1, bankroll: [100, 100] }));
  });
  await forceCalls(motion);
  await motion.goto(url);
  await motion.waitForSelector(".scene-ready");
  await motion.clock.runFor(2100);
  await motion.locator("#callButton").click(); // Check the big blind and deal the flop.
  await motion.clock.runFor(60);
  await motion.locator("#callButton").click(); // Check again while those cards are moving.
  await motion.clock.runFor(760);
  assert.equal(await motion.locator("#tableScene").getAttribute("data-animating"), "true", "The deal continues after the check gesture finishes");
  await screenshot(motion, "table-uninterrupted-deal");
  await motion.clock.runFor(450);
  assert.equal(await motion.locator("#tableScene").getAttribute("data-animating"), "false", "The original deal finishes on schedule");
  await motion.clock.runFor(1000);
  await motion.locator("#callButton").click(); // Turn.
  await motion.clock.runFor(2100);
  await motion.locator("#callButton").click(); // River and showdown.
  await motion.clock.runFor(2250);
  assert.equal(await motion.locator("#tableScene").getAttribute("data-animating"), "true");
  await screenshot(motion, "table-reveal-motion");
  await motion.locator("#cancelNextHandButton").click();
  await motion.locator("#newHandButton").click();
  await motion.clock.runFor(1500);
  assert.equal(await motion.locator("#tableScene").getAttribute("data-board-count"), "0");
  assert.equal(await motion.locator("#tableScene").getAttribute("data-revealed"), "false");
  assert.equal(await motion.locator("#tableScene").getAttribute("data-animating"), "false");
  await motion.close();

  // WebGL unavailable and context lost both leave a playable flat table.
  const fallback = await pageFor({ viewport: { width: 390, height: 844 } });
  await fallback.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...args) {
      return type.startsWith("webgl") ? null : getContext.call(this, type, ...args);
    };
  });
  await fallback.goto(url);
  await fallback.waitForSelector("#sceneNotice:not(.hidden)");
  await waitForMove(fallback);
  assert.equal(await fallback.locator(".scene-ready").count(), 0);
  assert.equal(await fallback.locator("#userCards .card").count(), 2);
  await assertFits(fallback);
  await screenshot(fallback, "table-fallback");
  await fallback.locator("#foldButton").click();
  await fallback.waitForSelector("#finishedActions:not(.hidden)");
  await fallback.close();

  const contextLoss = await pageFor();
  await contextLoss.clock.install({ time: new Date("2026-10-03T12:00:00Z") });
  await contextLoss.clock.pauseAt(new Date("2026-10-03T12:00:01Z"));
  await contextLoss.goto(url);
  await contextLoss.waitForSelector(".scene-ready");
  await waitForMove(contextLoss);
  await contextLoss.locator("#foldButton").click();
  await contextLoss.waitForSelector("#finishedActions:not(.hidden)");
  await contextLoss.emulateMedia({ reducedMotion: "reduce" });
  await contextLoss.clock.runFor(30);
  await contextLoss.waitForSelector('#tableScene[data-animating="false"]');
  await screenshot(contextLoss, "table-reduced-motion-reveal");
  await contextLoss.locator("#cancelNextHandButton").click();
  await contextLoss.locator("#tableCanvas").evaluate((canvas) => canvas.dispatchEvent(new Event("webglcontextlost", { cancelable: true })));
  await contextLoss.waitForSelector("#sceneNotice:not(.hidden)");
  assert.equal(await contextLoss.locator(".scene-ready").count(), 0);
  await contextLoss.close();
  assert.deepEqual(errors, [], "No uncaught browser errors");
  console.log("Browser checks passed: all streets, bets, showdown, fold, study, mobile, reduced motion, and WebGL fallback.");
} finally {
  await browser.close();
  server.close();
}
