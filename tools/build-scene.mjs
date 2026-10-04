import { build } from "esbuild";

await build({
  entryPoints: ["public/table-scene.js"],
  outfile: "public/table-scene.bundle.js",
  bundle: true,
  minify: true,
  format: "esm",
  target: "es2022",
  legalComments: "inline",
});
