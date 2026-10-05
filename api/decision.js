import { resolve } from "node:path";
import { HostedInference } from "../tools/hosted-inference.mjs";

export const inference = new HostedInference(resolve("artifacts/hosted"));

export default async function handler(request, response) {
  response.setHeader("Cache-Control", "no-store");
  if (request.method !== "POST") return response.status(405).json({ error:"Use POST" });
  try {
    if (Number(request.headers["content-length"] || 0) > 16384) throw new TypeError("Decision request too large");
    const body = typeof request.body === "string" ? JSON.parse(request.body) : request.body;
    return response.status(200).json(await inference.decision(body));
  } catch (error) {
    const invalid = error instanceof TypeError || error instanceof SyntaxError;
    if (!invalid) console.error("V3 native inference failed:", error);
    return response.status(invalid ? 400 : 503).json({ error:invalid ? error.message : "V3 is unavailable; try again shortly." });
  }
}
