import { resolve } from "node:path";
import { HostedArena } from "../tools/hosted-arena.mjs";

export const arena = new HostedArena(resolve("artifacts/hosted"));

export default async function handler(request, response) {
  response.setHeader("Cache-Control", "no-store");
  if (request.method !== "POST") return response.status(405).json({ error:"Use POST" });
  try {
    if (Number(request.headers["content-length"] || 0) > 4096) throw new TypeError("Arena request too large");
    const body = typeof request.body === "string" ? JSON.parse(request.body) : request.body;
    return response.status(200).json(await arena.match(body));
  } catch (error) {
    const invalid = error instanceof TypeError || error instanceof SyntaxError;
    if (!invalid && error.statusCode !== 409) console.error("Hosted arena failed:", error);
    return response.status(invalid ? 400 : error.statusCode === 409 ? 409 : 503)
      .json({ error:invalid || error.statusCode === 409 ? error.message : "Match could not finish. Try fewer hands." });
  }
}
