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

try {
  // Load the real model first, to catch import and initialization regressions.
  const smoke = await pageFor();
  await smoke.goto(url);
  await smoke.waitForSelector(".scene-ready");
  await smoke.waitForFunction(() => document.querySelectorAll("#userCards .card").length === 2);
  assert.equal(await smoke.locator("#agentCards .back").count(), 2);
  assert.equal(await smoke.locator("#modelStatus.ready").count(), 1);
  await smoke.waitForTimeout(700);
  await assertFits(smoke);
  await screenshot(smoke, "table-desktop");
  await smoke.close();

  // Force the trained action mix to check/call so every street is repeatable.
  const page = await pageFor();
  await page.addInitScript(() => { Math.random = () => .5; });
  for (const file of ["preflop-model.json", "postflop-model.json"]) {
    const nodes = JSON.parse(await readFile(`public/${file}`, "utf8"));
    for (const key of Object.keys(nodes)) nodes[key] = [0, 1, 0, 50_000];
    await page.route(`**/${file}`, (route) => route.fulfill({ json: nodes }));
  }
  await page.goto(url);
  await page.waitForSelector(".scene-ready");
  await waitForMove(page);
  assert.equal(await page.locator("#pot").textContent(), "1.5 BB");
  await page.locator("#callButton").click();
  assert.equal(await page.locator("#raiseSlider").isDisabled(), true, "Sizing must pause on the opponent's turn");
  await waitForMove(page);
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
      await screenshot(page, "table-small-phone");
      await page.setViewportSize({ width: 1440, height: 960 });
    }
    await page.locator("#callButton").click();
    if (street !== "River") await waitForMove(page);
  }
  await page.waitForSelector("#finishedActions:not(.hidden)");
  assert.equal(await page.locator("#agentCards .back").count(), 0);
  assert.equal(await page.locator("#tableScene").getAttribute("data-revealed"), "true");
  const stacks = await page.locator("#userStack, #agentStack").allTextContents();
  assert.equal(stacks.reduce((sum, text) => sum + parseFloat(text), 0), 200, "Showdown conserves chips");
  await page.locator("#cancelNextHandButton").click();
  assert.equal(await page.locator("#nextHandCountdown").textContent(), "Auto-deal cancelled.");
  await page.waitForTimeout(700);
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
  await page.locator("#cancelNextHandButton").click();
  await page.close();

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

  const contextLoss = await pageFor({ reducedMotion: "reduce" });
  await contextLoss.goto(url);
  await contextLoss.waitForSelector(".scene-ready");
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
