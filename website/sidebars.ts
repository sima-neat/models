import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  developerGuide: [
    {type: 'doc', id: 'README', label: 'Overview'},
    {
      type: 'category',
      label: 'Start here',
      collapsed: false,
      items: [
        {type: 'doc', id: 'getting-started', label: 'Getting started'},
        {type: 'doc', id: 'concepts', label: 'Concepts'},
        {type: 'doc', id: 'downloading-models', label: 'Download a model'},
      ],
    },
    {
      type: 'category',
      label: 'Contribute a model',
      collapsed: false,
      items: [
        {type: 'doc', id: 'adding-a-model', label: '1. Define the model contract'},
        {type: 'doc', id: 'local-development', label: '2. Qualify locally'},
        {type: 'doc', id: 'contributing', label: '3. Submit for review'},
        {type: 'doc', id: 'model-promotion', label: '4. Request promotion'},
      ],
    },
    {
      type: 'category',
      label: 'Data and evaluation',
      items: [
        {type: 'doc', id: 'datasets', label: 'Dataset access and registration'},
        {type: 'doc', id: 'local-accuracy-development', label: 'Accuracy evaluation'},
        {type: 'doc', id: 'benchmarking', label: 'DevKit benchmarking'},
      ],
    },
    {
      type: 'category',
      label: 'Contracts and reference',
      items: [
        'metadata-schema',
        'conversion-artifacts',
        {type: 'doc', id: 'model-readme', label: 'Model README and hero'},
        {type: 'doc', id: 'model-browser', label: 'Model browser'},
        'runner-class-requests',
        'model-taxonomy',
        'illustration-guide',
      ],
    },
  ],
};

export default sidebars;
