import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { unpackCheckpoint } from "./hosted-inference.mjs";

const execute = promisify(execFile);
const choices = new Set(["bundled", "v2", "v3", "baseline:random", "baseline:call", "baseline:pot"]);

export class HostedArena {
  constructor(directory, temporary = join(tmpdir(), "lard-arena")) {
    this.directory = directory;
    this.temporary = temporary;
  }
  async match(body) {
    const { a, b, hands = 10_000, seed = 1 } = body || {};
    if (!choices.has(a) || !choices.has(b)) throw new TypeError("Choose an available checkpoint or baseline");
    if (!Number.isInteger(hands) || hands < 2 || hands > 100_000 || hands % 2) {
      throw new TypeError("Choose an even hand count between 2 and 100,000");
    }
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new TypeError("Seed must be 0..4294967295");
    if (this.busy) {
      const error = new Error("An arena match is already running. Try again shortly.");
      error.statusCode = 409;
      throw error;
    }
    this.busy = true;
    try {
      if ([a, b].includes("v3") && !this.checkpoint) this.checkpoint = await unpackCheckpoint(this.directory, this.temporary);
      const source = (id) => id.startsWith("baseline:") ? id : id === "v3" ? this.checkpoint
        : join(this.directory, id === "bundled" ? "v1.policy" : "v2.bin");
      const binary = join(this.directory, `arena${process.platform === "win32" ? ".exe" : ""}`);
      const { stdout } = await execute(binary, ["--a", source(a), "--b", source(b), "--hands", String(hands),
        "--seed", String(seed), "--memory-mb", "1700",
        ...(a === "bundled" ? ["--swap-a-legacy-positions"] : []),
        ...(b === "bundled" ? ["--swap-b-legacy-positions"] : [])],
      { timeout:240_000, killSignal:"SIGKILL", maxBuffer:512 * 1024 });
      const report = JSON.parse(stdout);
      report.a.checkpoint = a;
      report.b.checkpoint = b;
      return report;
    } finally { this.busy = false; }
  }
}
