// sidebars.js

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'ROS2',
      items: ['ros2/introduction', 'ros2/nodes', 'ros2/topics'],
    },
    {
      type: 'category',
      label: 'Gazebo',
      items: ['gazebo/introduction', 'gazebo/physics', 'gazebo/simulation'],
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