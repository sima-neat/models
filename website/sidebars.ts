import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  developerGuide: [
    {type: 'doc', id: 'README', label: 'Overview'},
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
        {type: 'doc', id: 'model-browser', label: 'Model browser'},
        'runner-class-requests',
        'runner-integration',
        'model-taxonomy',
      ],
    },
  ],
};

export default sidebars;
