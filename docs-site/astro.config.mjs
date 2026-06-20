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
          autogenerate: { directory: 'architecture' },
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
          label: 'Design',
          autogenerate: { directory: 'design' },
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
