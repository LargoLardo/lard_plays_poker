import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, rename, rm } from "node:fs/promises";
import { spawn } from "node:child_process";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createInterface } from "node:readline";
import { pipeline } from "node:stream/promises";
import { createGunzip } from "node:zlib";

export function decisionLine(body) {
  const { nodeset, hero, board, history, actor, street } = body || {};
  const card = (value) => typeof value === "string" && /^[2-9TJQKA][shdc]$/.test(value);
  if (nodeset !== "v3" || !Array.isArray(hero) || hero.length !== 2 || !Array.isArray(board)
      || ![0, 1].includes(actor) || ![0, 1, 2, 3].includes(street)
      || board.length !== (street ? street + 2 : 0)) throw new TypeError("Invalid decision position");
  const visible = [...hero, ...board];
  if (!visible.every(card) || new Set(visible).size !== visible.length) throw new TypeError("Invalid or duplicate cards");
  if (!Array.isArray(history) || history.length > 40 || !history.every((event) => Array.isArray(event) && event.length === 3
      && [0, 1, 2].includes(event[0]) && Number.isFinite(event[1]) && event[1] >= 0 && event[1] <= 1000
      && [0, 1].includes(event[2]))) throw new TypeError("Invalid observed actions");
  return [...hero, board.length, ...board, history.length, ...history.flat(), actor, street].join(" ");
}

export class HostedInference {
  constructor(directory, temporary = join(tmpdir(), "lard-v3")) {
    this.directory = directory;
    this.temporary = temporary;
    this.queue = Promise.resolve();
  }
  close() {
    this.engine?.kill();
    this.lines?.close();
    this.engine = null;
  }
  async start() {
    if (this.engine && this.engine.exitCode === null && !this.engine.killed) return;
    if (!this.checkpoint) {
      await mkdir(this.temporary, { recursive:true });
      const checkpoint = join(this.temporary, "v3.bin");
      try {
        await pipeline(createReadStream(join(this.directory, "v3.bin.gz")), createGunzip(), createWriteStream(`${checkpoint}.tmp`));
        await rename(`${checkpoint}.tmp`, checkpoint);
      } catch (error) {
        await rm(`${checkpoint}.tmp`, { force:true });
        throw error;
      }
      this.checkpoint = checkpoint;
    }
    const nativeName = `train_v5${process.platform === "win32" ? ".exe" : ""}`;
    this.engine = spawn(join(this.directory, nativeName), ["--resume", this.checkpoint, "--infer",
      "--memory-mb", "1700", "--cache-mb", "8", "--max-nodes", "0"], { stdio:["pipe", "pipe", "inherit"] });
    this.lines = createInterface({ input:this.engine.stdout });
  }
  decision(body) {
    const line = decisionLine(body);
    // The C++ stdin protocol is serial; concurrent visitors share one warm model.
    const result = this.queue.then(async () => {
      await this.start();
      return new Promise((resolve, reject) => {
        const engine = this.engine, lines = this.lines;
        const finish = (error, value) => {
          clearTimeout(timer);
          lines.removeListener("line", reply);
          engine.removeListener("error", fail);
          engine.removeListener("exit", stopped);
          engine.stdin.removeListener("error", fail);
          if (error) { this.close(); reject(error); } else resolve(value);
        };
        const fail = (error) => finish(error);
        const stopped = () => finish(new Error("Native inference stopped"));
        const reply = (text) => {
          try {
            const report = JSON.parse(text);
            if (report.error) throw new TypeError(report.error);
            finish(null, report);
          } catch (error) { finish(error); }
        };
        const timer = setTimeout(() => finish(new Error("Native inference timed out")), 45_000);
        lines.once("line", reply);
        engine.once("error", fail);
        engine.once("exit", stopped);
        engine.stdin.once("error", fail);
        engine.stdin.write(`${line}\n`);
      });
    });
    this.queue = result.catch(() => {});
    return result;
  }
}
