import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const siteUrl = process.env.DOCS_SITE_URL?.trim() || 'https://crispy-dollop-km416j7.pages.github.io';
const basePath = process.env.DOCS_BASE_URL?.trim().replace(/^\/+|\/+$/g, '') ?? '';
const baseUrl = basePath ? `/${basePath}/` : '/';

const config: Config = {
  title: 'SiMa Neat Model Registry',
  tagline: 'Introduce, validate, and qualify models with reproducible evidence.',
  favicon: 'img/favicon.svg',

  future: {
    v4: true,
  },

  url: siteUrl,
  baseUrl,
  organizationName: 'sima-neat',
  projectName: 'models',
  trailingSlash: false,
  onBrokenLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/sima-neat/models/edit/develop/docs/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    metadata: [
      {
        name: 'description',
        content: 'Developer documentation for contributing models to the SiMa Neat Model Registry.',
      },
    ],
    navbar: {
      title: 'Model Registry',
      logo: {
        alt: 'SiMa Neat Model Registry mark',
        src: 'img/mark.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'developerGuide',
          position: 'left',
          label: 'Developer guide',
        },
        {
          to: '/docs/metadata-schema',
          label: 'YAML reference',
          position: 'left',
        },
        {
          href: 'https://github.com/sima-neat/models',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Start here',
          items: [
            {label: 'Add a model', to: '/docs/adding-a-model'},
            {label: 'Local testing', to: '/docs/local-development'},
            {label: 'Contribution checklist', to: '/docs/contributing'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'model.yaml', to: '/docs/metadata-schema'},
            {label: 'Taxonomy', to: '/docs/model-taxonomy'},
            {label: 'Architecture', to: '/docs/architecture'},
          ],
        },
        {
          title: 'Source',
          items: [
            {label: 'Models repository', href: 'https://github.com/sima-neat/models'},
            {label: 'Report a documentation issue', href: 'https://github.com/sima-neat/models/issues/new'},
          ],
        },
      ],
      copyright: `SiMa Neat Model Registry · ${new Date().getFullYear()}`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'python', 'yaml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
