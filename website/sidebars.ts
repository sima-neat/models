import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  developerGuide: [
    {type: 'doc', id: 'README', label: 'Overview'},
    {
      type: 'category',
      label: 'Contribute a model',
      collapsed: false,
      items: ['adding-a-model', 'local-development', 'contributing'],
    },
    {
      type: 'category',
      label: 'Contracts and reference',
      items: ['metadata-schema', 'model-taxonomy', 'benchmarking'],
    },
  ],
};

export default sidebars;
