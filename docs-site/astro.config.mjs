import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// The capability cards moved from benchmarks/<machine>-mtl5-bench/ to
// benchmarks/cards/<machine>/ (#799); the old URLs keep working.
const CARD_MACHINES = [
  'arduino-ventuno-q', 'i7-12700k', 'jetson-orin', 'ryzen-7-2700x',
  'ryzen-7-8845hs', 'ryzen-9-8945hs', 'xeon-e5-2420v2',
];

export default defineConfig({
  site: 'https://stillwater-sc.github.io',
  base: '/mtl5',
  redirects: Object.fromEntries(CARD_MACHINES.map((m) => [
    `/benchmarks/${m}-mtl5-bench`, `/mtl5/benchmarks/cards/${m}/`,
  ])),
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: 'MTL5 -- Matrix Template Library',
      description:
        'A C++20 header-only linear algebra library for mixed-precision algorithm design',
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/stillwater-sc/mtl5',
        },
      ],
      // Turns the edit link on. Every page sets its own editUrl in
      // sync-content.mjs, because Starlight would join this base with the
      // generated src/content/docs/ path.
      editLink: {
        baseUrl:
          'https://github.com/stillwater-sc/mtl5/edit/main/docs/',
      },
      customCss: [
        'katex/dist/katex.min.css',
        './src/styles/custom.css',
      ],
      sidebar: [
        {
          label: 'Getting Started',
          autogenerate: { directory: 'getting-started' },
        },
        {
          label: 'Architecture',
          items: [
            { slug: 'architecture' },
            { slug: 'architecture/concepts' },
            { slug: 'architecture/aggregate-types' },
            { slug: 'architecture/accuracy-architecture' },
            { slug: 'architecture/exact-accumulator' },
            { slug: 'architecture/error-free-transformations' },
            {
              // Per-type architecture docs for every mat/ and vec/ container.
              label: 'Containers',
              autogenerate: { directory: 'architecture/containers' },
            },
          ],
        },
        {
          label: 'Modernization',
          autogenerate: { directory: 'modernization' },
        },
        {
          label: 'Linear Algebra Algorithms',
          autogenerate: { directory: 'algorithms' },
        },
        {
          // Where an implementation choice (summation order, contraction)
          // decides the bits, and how a caller controls it (#556).
          label: 'Numerical Considerations',
          items: [
            { slug: 'numerical' },
            { slug: 'numerical/summation-order-in-cholesky' },
            { slug: 'numerical/householder-reflector-forms' },
          ],
        },
        {
          // Analyses first, the generated capability cards last (#799). A new
          // page in docs/benchmarks/ needs an entry here; a new card does not:
          // publish_platform.cmake writes it to benchmarks/cards/.
          label: 'Benchmarking',
          items: [
            { slug: 'benchmarks' },
            { slug: 'benchmarks/systems' },
            { slug: 'benchmarks/run-contract' },
            {
              label: 'Across the machines',
              items: [
                { slug: 'benchmarks/cache-blocking-ab' },
                { slug: 'benchmarks/register-tile-sweep' },
                { slug: 'benchmarks/interleave-width' },
                { slug: 'benchmarks/direct-solver-references' },
              ],
            },
            {
              label: 'Per-machine analyses',
              items: [
                { slug: 'benchmarks/i7-12700k' },
                { slug: 'benchmarks/ryzen-9-8945hs' },
                { slug: 'benchmarks/jetson-orin' },
                { slug: 'benchmarks/arduino-ventuno-q' },
              ],
            },
            {
              label: 'Capability cards',
              collapsed: true,
              autogenerate: { directory: 'benchmarks/cards' },
            },
          ],
        },
        {
          label: 'Design',
          items: [
            {
              label: 'Parallelization',
              items: [
                { slug: 'design/parallelization-patterns-and-pitfalls' },
                { slug: 'design/concurrency-runtime-ownership' },
                { slug: 'design/blas-kernel-architecture' },
              ],
            },
            {
              label: 'Library',
              items: [
                { slug: 'design/mixed-precision-acceleration' },
                { slug: 'design/expression-template-architecture' },
                { slug: 'design/operation-dispatch-architecture' },
                { slug: 'design/iterative-solvers-architecture' },
                { slug: 'design/eigensolvers-architecture' },
                { slug: 'design/smoothers-architecture' },
                { slug: 'design/multigrid-architecture' },
                { slug: 'design/mixed-precision-custom-types-simd' },
                { slug: 'design/dense-direct-solvers-architecture' },
                { slug: 'design/sparse-direct-solvers' },
              ],
            },
          ],
        },
        {
          // Lifted out of Design: these are experiments and their results, not
          // architecture. They answer "what did the machine do", which is a
          // different question from "how is it built", and they are cited from
          // issues and PRs often enough to deserve a top-level home.
          label: 'Performance studies',
          items: [
            { slug: 'performance/benchmarking-methodology' },
            { slug: 'performance/blocking-and-scheduling-assessment' },
            { slug: 'performance/mc-critical-panels' },
            { slug: 'performance/hardware-expansion-plan' },
            { slug: 'performance/arm64-simd-neon-sve' },
            { slug: 'performance/cache-blocking-ab-study' },
            { slug: 'performance/ikj-gemm-and-crs-dense' },
            { slug: 'performance/multicore-scaling-investigation' },
            { slug: 'performance/issue-297-threading-benchmark-plan' },
            { slug: 'performance/issue-297-threading-results' },
          ],
        },
        {
          label: 'Examples',
          autogenerate: { directory: 'examples' },
        },
        {
          label: 'Generators',
          autogenerate: { directory: 'generators' },
        },
        {
          label: 'Contributing',
          autogenerate: { directory: 'contributing' },
        },
        {
          // The peer sites: Universal's number types, and the mp-numerics
          // studies that run MTL5's algorithms in them.
          label: 'Related',
          items: [
            {
              label: 'mp-numerics studies',
              link: 'https://stillwater-sc.github.io/mp-numerics/',
            },
            {
              label: 'Universal number systems',
              link: 'https://stillwater-sc.github.io/universal/',
            },
          ],
        },
        {
          // Doxygen C++ API reference, generated into public/api/ by
          // `npm run api` and served at <base>api/ (i.e. /mtl5/api/ on
          // Pages, /api/ in local dev). Starlight prepends the site `base` to
          // internal links, so this stays base-relative. Opens in a new tab
          // since it is a separate, fully self-contained doc tree.
          label: 'C++ API Reference (Doxygen)',
          link: '/api/',
          attrs: { target: '_blank', rel: 'noopener noreferrer' },
        },
      ],
    }),
  ],
});
