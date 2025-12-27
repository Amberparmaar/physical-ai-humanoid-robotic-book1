// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'An AI-Native Textbook',
  favicon: 'img/logo.png',

  // Set the production url of your site here
  url: 'https://amberparmaar.github.io',
  // Set the /<base>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<org-name>/'
  baseUrl: '/physical-ai-humanoid-robotic-book1/',

  // GitHub pages deployment config.
  organizationName: 'amberparmaar', // Usually your GitHub org/user name.
  projectName: 'physical-ai-humanoid-robotic-book1', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ur'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Amberparmaar/physical-ai-humanoid-robotic-book1/edit/main/',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/Amberparmaar/physical-ai-humanoid-robotic-book1/edit/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themes: [
    // ... other themes
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Physical AI & Robotics Logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Course Textbook',
          },
          {
            to: '/courses',
            label: 'Course Modules',
            position: 'left',
          },
          {
            to: '/blog',
            label: 'Blog',
            position: 'left',
          },
          {
            href: 'https://github.com/Amberparmaar/physical-ai-humanoid-robotic-book1',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Course Content',
            items: [
              {
                label: 'Course Introduction',
                to: '/docs/intro',
              },
              {
                label: 'ROS2 Fundamentals',
                to: '/docs/ros2/introduction',
              },
              {
                label: 'Gazebo Simulation',
                to: '/docs/gazebo/introduction',
              },
              {
                label: 'NVIDIA Isaac Platform',
                to: '/docs/isaac/introduction',
              },
              {
                label: 'VLA Models',
                to: '/docs/vla/introduction',
              },
            ],
          },
          {
            title: 'Learning Tools',
            items: [
              {
                label: 'Course Modules',
                to: '/courses',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/Amberparmaar/physical-ai-humanoid-robotic-book1',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
