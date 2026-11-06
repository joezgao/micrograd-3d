#!/bin/bash
# Build, serve the production bundle, and run Lighthouse (mobile + desktop).
# Prints one score line per form factor; full reports go to .factory/logs/lighthouse-*.json.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .factory/logs
npm run -s build > .factory/logs/lh-build.log 2>&1
npx vite preview --port 4173 --strictPort > .factory/logs/lh-preview.log 2>&1 &
PREVIEW=$!
trap 'kill $PREVIEW 2>/dev/null' EXIT
for _ in $(seq 1 40); do curl -sf http://localhost:4173 >/dev/null && break; sleep 0.25; done
for ff in mobile desktop; do
  preset=(); [ "$ff" = desktop ] && preset=(--preset=desktop)
  npx lighthouse http://localhost:4173 ${preset[@]+"${preset[@]}"} --quiet --output=json \
    --output-path=".factory/logs/lighthouse-$ff.json" \
    --chrome-flags="--headless=new" >/dev/null 2>&1
  node -e '
    const r = require(process.argv[1]); const c = r.categories;
    const s = (k) => Math.round(c[k].score * 100);
    console.log(`${process.argv[2].padEnd(8)} performance ${s("performance")} · accessibility ${s("accessibility")} · best-practices ${s("best-practices")} · seo ${s("seo")}`);
  ' "$PWD/.factory/logs/lighthouse-$ff.json" "$ff"
done
