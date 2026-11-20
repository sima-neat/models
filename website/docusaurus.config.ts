import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import {portalConfig} from './scripts/portal-config.mjs';

const portal = portalConfig();
const siteUrl = process.env.DOCS_SITE_URL?.trim() || portal.siteUrl;
const basePath = process.env.DOCS_BASE_URL?.trim().replace(/^\/+|\/+$/g, '') ?? '';
const baseUrl = basePath ? `/${basePath}/` : '/';

const config: Config = {
  title: 'SiMa Neat Model Browser',
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
  onBrokenAnchors: 'throw',
  storage: {
    type: 'localStorage',
    namespace: false,
  },
  clientModules: ['./src/clientModules/developer-portal.js', './src/clientModules/analytics.js'],
  customFields: {
    analytics: portal.analytics,
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
        content: 'Developer documentation for contributing models to the SiMa Neat Model Browser.',
      },
    ],
    navbar: {
      title: 'Model Browser',
      logo: {
        alt: 'SiMa Neat Model Browser mark',
        src: 'img/mark.svg',
      },
      items: [
        {
          type: 'html',
          position: 'left',
          className: 'navbar__catalog-item',
          value: `<a class="navbar__link" href="${baseUrl}">Browse models</a>`,
        },
        {
          type: 'html',
          position: 'left',
          className: 'navbar__catalog-item',
          value: `<a class="navbar__link" href="${baseUrl}?view=compare">Compare</a>`,
        },
        ...(portal.uiMode === 'developer' ? [{
          type: 'docSidebar' as const,
          sidebarId: 'developerGuide',
          position: 'left' as const,
          label: 'Developer docs',
        }, {
          type: 'html' as const,
          position: 'right' as const,
          className: 'navbar__environment-item',
          value: '<span class="navbar__environment"><i></i>Staging registry</span>',
        }] : []),
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Start here',
          items: [
            {label: 'Getting started', to: '/docs/getting-started'},
            {label: 'Add a model', to: '/docs/adding-a-model'},
            {label: 'Local testing', to: '/docs/local-development'},
            {label: 'Contribution checklist', to: '/docs/contributing'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'model.yaml', to: '/docs/metadata-schema'},
            {label: 'Datasets', to: '/docs/datasets'},
            {label: 'Taxonomy', to: '/docs/model-taxonomy'},
          ],
        },
        {
          title: 'Source',
          items: [
            {label: 'Models repository', href: 'https://github.com/sima-neat/models'},
            {label: 'Dataset Registry', href: 'https://github.com/sima-neat/dataset'},
            {label: 'Report a documentation issue', href: 'https://github.com/sima-neat/models/issues/new'},
            {html: '<button type="button" class="footer__link-item cookie-preferences-link" data-cookie-preferences>Cookie preferences</button>'},
          ],
        },
      ],
      copyright: `SiMa Neat Model Browser · ${new Date().getFullYear()}`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'json', 'python', 'yaml'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
