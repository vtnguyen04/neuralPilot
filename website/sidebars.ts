import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Sidebars configuration for NeuroPilot documentation.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    'getting-started',
    {
      type: 'category',
      label: 'API & CLI Reference',
      items: [
        'api/cli-reference',
        'api/python-sdk',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Neural Architecture',
      items: [
        'architecture/backbones',
        'architecture/neck-routing',
        'architecture/task-heads',
        'architecture/loss-functions',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Data Engineering',
      items: [
        'data/custom-datasets',
        'data/video-pipelines',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Edge Deployment',
      items: [
        'deploy/export-onnx-trt',
        'deploy/jetson-inference',
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Advanced Customization',
      items: [
        'advanced/customization',
      ],
      collapsed: false,
    },
  ],
};

export default sidebars;
