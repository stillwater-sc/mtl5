import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://stillwater-sc.github.io',
  base: '/mtl5',
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
          label: 'Benchmarking',
          autogenerate: { directory: 'benchmarks' },
        },
        {
          label: 'Design',
          items: [
            {
              label: 'Parallelization',
              items: [
                { slug: 'design/parallelization-patterns-and-pitfalls' },
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
            { slug: 'performance/cache-blocking-ab-study' },
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
