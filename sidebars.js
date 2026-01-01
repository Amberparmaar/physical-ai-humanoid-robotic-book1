// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'ROS2',
      items: ['ros2/introduction', 'ros2/nodes', 'ros2/topics'],
    },
    {
      type: 'category',
      label: 'Gazebo & Unity',
      items: ['gazebo/introduction', 'gazebo/simulation', 'gazebo/physics'],
    },
    {
      type: 'category',
      label: 'NVIDIA Isaac',
      items: ['isaac/introduction', 'isaac/perception', 'isaac/control'],
    },
    {
      type: 'category',
      label: 'VLA Models',
      items: ['vla/introduction', 'vla/vision', 'vla/language', 'vla/action'],
    },
    {
      type: 'category',
      label: 'AI Robot Brain',
      items: [
        'ai-robot-brain/intro',
        'ai-robot-brain/chapter-2-perception-systems-computer-vision',
        'ai-robot-brain/chapter-3-planning-decision-making',
        'ai-robot-brain/chapter-4-control-systems-motor-skills',
        'ai-robot-brain/chapter-5-learning-adaptation',
        'ai-robot-brain/chapter-6-human-robot-interaction',
        'ai-robot-brain/chapter-7-cognitive-architectures-integration'
      ],
    },
    {
      type: 'category',
      label: 'Robotic Nervous System (ROS2)',
      items: [
        'robotic-nervous-system/intro',
        'robotic-nervous-system/chapter-2-ros2-architecture',
        'robotic-nervous-system/chapter-3-nodes-communication',
        'robotic-nervous-system/chapter-4-topics-services-actions',
        'robotic-nervous-system/chapter-5-parameters-configuration',
        'robotic-nervous-system/chapter-6-advanced-communication',
        'robotic-nervous-system/chapter-7-real-world-applications'
      ],
    },
    {
      type: 'category',
      label: 'Digital Twin & Simulation',
      items: [
        'digital-twin/intro',
        'digital-twin/module-overview',
        'digital-twin/chapter-2-gazebo-fundamentals',
        'digital-twin/chapter-3-robot-modeling-urdf',
        'digital-twin/chapter-4-sensor-simulation-perception',
        'digital-twin/chapter-5-unity-robotics-simulation',
        'digital-twin/chapter-6-simulation-to-reality-transfer',
        'digital-twin/chapter-7-advanced-applications-case-studies',
        'digital-twin/chapter-9-conclusion-future-directions'
      ],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action Models',
      items: [
        'vision-language-action/intro',
        'vision-language-action/module-overview',
        'vision-language-action/chapter-2-foundations-multimodal-learning',
        'vision-language-action/chapter-3-vision-processing-scene-understanding',
        'vision-language-action/chapter-4-language-understanding-grounding',
        'vision-language-action/chapter-5-action-generation-control',
        'vision-language-action/chapter-6-multimodal-integration-architecture',
        'vision-language-action/chapter-7-advanced-applications-case-studies',
        'vision-language-action/chapter-8-conclusion-future-directions'
      ],
    },
    {
      type: 'category',
      label: 'Control Systems',
      items: [
        'control-systems/intro',
        'control-systems/module-overview'
      ],
    },
  ],
};

module.exports = sidebars;