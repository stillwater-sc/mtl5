#!/usr/bin/env node
/**
 * Syncs ALL documentation content from docs/ into Starlight's
 * src/content/docs/ tree.
 *
 * ARCHITECTURAL RULE: src/content/docs/ is 100% GENERATED.
 * ─────────────────────────────────────────────────────────
 * Every page — whether "hand-written" or transformed from repo docs —
 * originates in docs/.  Nothing is ever authored directly in
 * src/content/docs/.  The entire directory is .gitignored.
 *
 * This eliminates the confusion where Claude Code (or humans) write
 * content into docs-site/src/content/docs/ instead of docs/, only to
 * have it ignored by .gitignore or overwritten by this script.
 *
 * Content categories:
 *   1. SITE pages  (docs/site/*.mdx) — copied verbatim (already have frontmatter)
 *   2. SYNCED docs (docs/**\/*.md)   — H1 extracted as title, frontmatter added,
 *                                      image paths & links rewritten
 *   3. ROOT files  (CHANGELOG.md, etc.) — same transform as synced docs
 *
 * Run automatically via `npm run build` / `npm run dev`.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, cpSync, rmSync } from 'fs';
import { dirname, join, posix } from 'path';

const REPO = join(import.meta.dirname, '..');
const DOCS = join(REPO, 'docs');
const OUT  = join(import.meta.dirname, 'src', 'content', 'docs');
const BASE = '/mtl5';  // Astro base path

// ── Site pages (MDX with Starlight components) ────────────────────
// These are copied verbatim — they already contain frontmatter.
const SITE_FILES = {
  'site/index.mdx': 'index.mdx',
};

// ── Synced docs (source path relative to docs/ → dest relative to content/docs/) ──
const FILE_MAP = {
  // ── Getting Started ─────────────────────────────────────────────
  'getting-started/index.md':        'getting-started/index.md',
  'getting-started/build-options.md': 'getting-started/build-options.md',

  // ── Architecture ────────────────────────────────────────────────
  'architecture/index.md':           'architecture/index.md',
  'architecture/concepts.md':        'architecture/concepts.md',

  // ── Design ──────────────────────────────────────────────────────
  'sparse-direct-solvers-design.md':            'design/sparse-direct-solvers.md',
  'position-mixed-precision-acceleration.md':   'design/mixed-precision-acceleration.md',

  // ── Modernization ──────────────────────────────────────────────────────
  'modernization/advanced-itl-components.md':                  'modernization/advanced-itl-components.md',
  'modernization/mtl4-modernization.md':                       'modernization/mtl4-modernization.md',
  'modernization/element-wise-transcendental-functions.md':    'modernization/element-wise-transcendental-functions.md',
  'modernization/examples.md':                                 'modernization/examples.md',
  'modernization/vector-and-insertion-utilities.md':           'modernization/vector-and-insertion-utilities.md',

  // ── Examples ────────────────────────────────────────────────────
  'examples/expression-templates.md':  'examples/expression-templates.md',
  'examples/numerical-examples.md':    'examples/numerical-examples.md',

  // ── Generators ──────────────────────────────────────────────────
  'generators/strengthen-LA-tests-with-generator-matrices.md': 'generators/strengthen-tests.md',
  'generators/tier1-2-matrices.md':                            'generators/tier1-2-matrices.md',
  'generators/tier3-spectral-control.md':                      'generators/tier3-spectral-control.md',
};

// ── Root files (relative to repo root) ────────────────────────────
const ROOT_FILE_MAP = {
  'CHANGELOG.md': 'changelog.md',
};

// ── Link lookup ───────────────────────────────────────────────────

function buildLinkLookup() {
  const lookup = {};
  for (const [src, dest] of Object.entries(FILE_MAP)) {
    const slug = dest.replace(/\.md$/, '').replace(/\/index$/, '/');
    lookup[src] = `${BASE}/${slug.endsWith('/') ? slug : slug + '/'}`;
  }
  for (const [src, dest] of Object.entries(ROOT_FILE_MAP)) {
    const slug = dest.replace(/\.md$/, '').replace(/\/index$/, '/');
    lookup[`../${src}`] = `${BASE}/${slug.endsWith('/') ? slug : slug + '/'}`;
  }
  return lookup;
}

const LINK_LOOKUP = buildLinkLookup();

// ── Transforms ────────────────────────────────────────────────────

function rewriteLinks(content, srcRelative) {
  const srcDir = posix.dirname(srcRelative);
  return content.replace(/\]\(([^)]+\.md)\)/g, (match, target) => {
    if (target.startsWith('http://') || target.startsWith('https://')) return match;
    const resolved = posix.normalize(posix.join(srcDir, target));
    const url = LINK_LOOKUP[resolved];
    return url ? `](${url})` : match;
  });
}

function extractTitle(content) {
  const match = content.match(/^#\s+(.+)$/m);
  return match ? match[1].trim() : 'Untitled';
}

function stripFirstHeading(content) {
  return content.replace(/^#\s+.+\n*/m, '');
}

function rewriteImagePaths(content) {
  return content
    .replace(/\]\(img\//g, `](${BASE}/img/`)
    .replace(/```bib/g, '```text');
}

function addFrontmatter(content, srcRelative) {
  const title = extractTitle(content);
  let body = stripFirstHeading(content);
  body = rewriteImagePaths(body);
  body = rewriteLinks(body, srcRelative);
  const safeTitle = title.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `---\ntitle: "${safeTitle}"\n---\n\n${body}`;
}

// ── File operations ───────────────────────────────────────────────

function writeOut(destRelative, content) {
  const destPath = join(OUT, destRelative);
  mkdirSync(dirname(destPath), { recursive: true });
  writeFileSync(destPath, content);
}

function syncMarkdown(srcPath, srcRelative, destRelative) {
  if (!existsSync(srcPath)) {
    console.warn(`  SKIP (not found): ${srcPath}`);
    return;
  }
  const content = readFileSync(srcPath, 'utf-8');
  writeOut(destRelative, addFrontmatter(content, srcRelative));
}

function copySitePage(srcPath, destRelative) {
  if (!existsSync(srcPath)) {
    console.warn(`  SKIP (not found): ${srcPath}`);
    return;
  }
  const content = readFileSync(srcPath, 'utf-8');
  writeOut(destRelative, content);
}

// ── Main ──────────────────────────────────────────────────────────

// Clear stale Astro data store cache
const astroCache = join(import.meta.dirname, 'node_modules', '.astro');
if (existsSync(astroCache)) {
  rmSync(astroCache, { recursive: true });
}

// Wipe the entire output directory — it's 100% generated
if (existsSync(OUT)) {
  rmSync(OUT, { recursive: true });
}
mkdirSync(OUT, { recursive: true });

console.log('Syncing docs/ → docs-site/src/content/docs/ ...');

// 1. Copy site pages (MDX with components, already have frontmatter)
for (const [src, dest] of Object.entries(SITE_FILES)) {
  copySitePage(join(DOCS, src), dest);
  console.log(`  site: ${src} → ${dest}`);
}

// 2. Sync docs/ markdown files (add frontmatter, rewrite links)
for (const [src, dest] of Object.entries(FILE_MAP)) {
  syncMarkdown(join(DOCS, src), src, dest);
  console.log(`  sync: ${src} → ${dest}`);
}

// 3. Sync repo-root files
for (const [src, dest] of Object.entries(ROOT_FILE_MAP)) {
  syncMarkdown(join(REPO, src), `../${src}`, dest);
  console.log(`  root: ${src} → ${dest}`);
}

// 4. Copy images to public/ (served as static assets at /mtl5/img/)
const PUB = join(import.meta.dirname, 'public');
const imgSrc = join(DOCS, 'img');
const imgDest = join(PUB, 'img');
if (existsSync(imgSrc)) {
  mkdirSync(imgDest, { recursive: true });
  cpSync(imgSrc, imgDest, { recursive: true });
  console.log('  Copied docs/img/ → public/img/');
}

console.log('Done.');
