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
  ],
};

module.exports = sidebars;