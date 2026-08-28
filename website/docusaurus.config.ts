import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const siteUrl = process.env.DOCS_SITE_URL?.trim() || 'http://localhost:4173';
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
  // Vulcan's /models CloudFront behavior rewrites clean routes to
  // <route>/index.html, so the static artifact must use directory indexes.
  trailingSlash: true,
  onBrokenLinks: 'throw',
  storage: {
    type: 'localStorage',
    namespace: false,
  },

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
          type: 'html',
          position: 'left',
          className: 'navbar__catalog-item',
          value: `<a class="navbar__link" href="${baseUrl}">Models</a>`,
        },
        {
          type: 'docSidebar',
          sidebarId: 'developerGuide',
          position: 'left',
          label: 'Developer docs',
        },
        {
          type: 'html',
          position: 'right',
          className: 'navbar__environment-item',
          value: '<span class="navbar__environment"><i></i>Staging registry</span>',
        },
      ],
    },
    footer: {
      style: 'light',
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
