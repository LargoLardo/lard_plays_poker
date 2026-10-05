import { createReadStream, createWriteStream } from "node:fs";
import { chmod, copyFile, mkdir, open, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createGzip } from "node:zlib";

const hosted = "artifacts/hosted";
const nativeName = `train_v5${process.platform === "win32" ? ".exe" : ""}`;
await mkdir(hosted, { recursive:true });
const python = process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
const catalog = [];
for (const [id, engine, magic] of [["v2", "v4", "LARDCPP4"], ["v3", "v5", "LARDCPP5"]]) {
  const source = `checkpoints/${id}.bin`;
  let file = await open(source);
  let header = Buffer.alloc(40);
  await file.read(header, 0, 40, 0);
  await file.close();
  if (header.subarray(0, 8).toString() !== magic) {
    // Vercel may check out LFS pointers. Fetch the pinned, public GitHub object.
    const pointer = await readFile(source, "utf8");
    const oid = pointer.match(/^oid sha256:([a-f0-9]{64})$/m)?.[1];
    const size = Number(pointer.match(/^size (\d+)$/m)?.[1]);
    if (!oid || !size) throw new Error(`Invalid checkpoint: ${source}`);
    const ref = process.env.VERCEL_GIT_COMMIT_SHA || "main";
    const response = await fetch(`https://media.githubusercontent.com/media/LargoLardo/lard_plays_poker/${ref}/${source}`);
    if (!response.ok || !response.body) throw new Error(`Could not download ${source}: ${response.status}`);
    const temporary = `${source}.download`;
    await pipeline(Readable.fromWeb(response.body), createWriteStream(temporary));
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(temporary)) hash.update(chunk);
    if ((await stat(temporary)).size !== size || hash.digest("hex") !== oid) {
      await rm(temporary);
      throw new Error(`Checkpoint checksum mismatch: ${source}`);
    }
    await rename(temporary, source);
    file = await open(source);
    await file.read(header, 0, 40, 0);
    await file.close();
    if (header.subarray(0, 8).toString() !== magic) throw new Error(`Wrong checkpoint version: ${source}`);
  }
  const iterations = Number(header.readBigUInt64LE(24));
  const totalNodes = Number(header.readBigUInt64LE(32));
  const directory = `public/models/${id}`;
  const temporary = `${hosted}/${id}-export.bin`;
  execFileSync(python, ["cpp/run.py", "--model", engine, "--resume", source,
    "--iterations", "0", "--memory-mb", "2048", "--output", temporary, "--export", directory,
    ...(engine === "v5" ? ["--cache-mb", "0", "--max-nodes", "0"] : [])], { stdio:"inherit" });
  await rm(temporary);
  catalog.push({ id, label:`${id.toUpperCase()} · ${Number((iterations / 1_000_000).toFixed(3))}M`,
    iterations, totalNodes, native:engine === "v5",
    preflop:`/models/${id}/preflop-model.json`, postflop:`/models/${id}/postflop-model.json` });
}
await copyFile(`cpp/build/${nativeName}`, `${hosted}/${nativeName}`);
await chmod(`${hosted}/${nativeName}`, 0o755);
await pipeline(createReadStream("checkpoints/v3.bin"), createGzip(), createWriteStream(`${hosted}/v3.bin.gz`));
// Keep the native function below Vercel's standard 250 MiB package ceiling.
if ((await stat(`${hosted}/v3.bin.gz`)).size + (await stat(`${hosted}/${nativeName}`)).size > 240 * 1024 * 1024) {
  throw new Error("The hosted V3 checkpoint exceeds the function package budget.");
}
await writeFile("public/nodesets.json", JSON.stringify(catalog));
console.log(`Hosted models ready: V1 · 10M, ${catalog.map((item) => item.label).join(", ")}.`);
