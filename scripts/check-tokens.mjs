// Design-system guard, copied from joe-site (vault: Claude Hub/设计系统/设计系统.md §5, §10 "Enforce, don't just document").
// Fails when CSS uses a raw colour, an off-scale spacing value or an off-scale radius.
//   colours: only inside custom-property definitions (`--name: #hex;`). Canvas/three.js colours in .tsx are data, not checked.
//   spacing: px in gap / padding / margin must be on the scale; em, %, var(), calc() and clamp() are not checked.
//   radius:  0 (texture, data), 12px (panels), 999px (buttons), 50% (dots).
//   escape:  a deliberate exception carries `/* token-ok: <reason> */` on the same line.
// Usage: node scripts/check-tokens.mjs  -> prints violations, exits 1 if any.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const SPACE = new Set([0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]);
const RADIUS = new Set(['0', '0px', '12px', '999px', '50%']);
const COLOR = /#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?|oklch|lab|lch)\(/;
const TOKEN_DEF = /^\s*--[a-z0-9-]+\s*:/;

const files = ['index.html'];
(function walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p);
    else if (name.endsWith('.css')) files.push(p);
  }
})('src');

const bad = [];
for (const file of files) {
  let text = readFileSync(file, 'utf8');
  if (file.endsWith('.html')) text = [...text.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((m) => m[1]).join('\n');
  text.split('\n').forEach((raw, i) => {
    if (/token-ok:/.test(raw)) return;
    const at = `${file}:${i + 1}`;
    // One line can hold several declarations (minified rules), so check each one.
    for (const decl of raw.replace(/\/\*.*?\*\//g, '').split(/[;{}]/)) {
      if (TOKEN_DEF.test(decl)) continue;
      const m = decl.match(/^\s*([a-z-]+)\s*:\s*(.+)$/);
      if (!m) continue;
      const [, prop, value] = m;
      if (COLOR.test(value)) bad.push(`${at}  ${prop}: raw colour; use a token (var(--…)): ${value.trim()}`);
      if (/^((row-|column-)?gap|padding(-[a-z]+)*|margin(-[a-z]+)*)$/.test(prop)) {
        for (const [, n] of value.replace(/\b(calc|clamp|max|min)\([^)]*\)/g, '').matchAll(/(-?\d*\.?\d+)px/g)) {
          if (!SPACE.has(Math.abs(Number(n)))) bad.push(`${at}  ${prop} ${n}px is off the 4·8·12·16·24·32·48·64·96·128 scale`);
        }
      }
      if (/^border(-[a-z]+)*-radius$/.test(prop)) {
        for (const v of value.trim().split(/\s+/)) {
          if (!RADIUS.has(v) && !/^var\(/.test(v)) bad.push(`${at}  ${prop} ${v} is not 0 / 12px / 999px / 50%`);
        }
      }
    }
  });
}

for (const b of bad) console.log(b);
console.log(`check-tokens: ${files.length} files, ${bad.length} violation(s)`);
process.exit(bad.length ? 1 : 0);
