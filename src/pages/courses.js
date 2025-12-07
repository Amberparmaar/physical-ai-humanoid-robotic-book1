import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './courses.module.css';

function CourseCard({title, description, duration, level, link}) {
  return (
    <div className={clsx('col col--4', styles.courseCard)}>
      <div className="card">
        <div className="card__header">
          <h3>{title}</h3>
        </div>
        <div className="card__body">
          <p>{description}</p>
          <div className={styles.courseInfo}>
            <span className={styles.duration}>⏱️ {duration}</span>
            <span className={styles.level}>🎓 {level}</span>
          </div>
        </div>
        <div className="card__footer">
          <Link to={link} className="button button--primary button--block">
            Start Learning
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function Courses() {
  const {siteConfig} = useDocusaurusContext();
  
  const courses = [
    {
      title: 'ROS2 Fundamentals',
      description: 'Learn the core concepts of ROS2, including nodes, topics, services, and actions for robotic applications.',
      duration: '4 weeks',
      level: 'Beginner',
      link: '/docs/ros2/introduction'
    },
    {
      title: 'Gazebo Simulation',
      description: 'Master robot simulation using Gazebo, including physics modeling, sensor simulation, and environment creation.',
      duration: '3 weeks',
      level: 'Intermediate',
      link: '/docs/gazebo/introduction'
    },
    {
      title: 'NVIDIA Isaac Platform',
      description: 'Explore GPU-accelerated robotics with NVIDIA Isaac, including perception, navigation, and manipulation.',
      duration: '5 weeks',
      level: 'Advanced',
      link: '/docs/isaac/introduction'
    },
    {
      title: 'Vision-Language-Action Models',
      description: 'Understand VLA models that combine perception, language understanding, and robotic action for humanoid robots.',
      duration: '6 weeks',
      level: 'Advanced',
      link: '/docs/vla/introduction'
    },
    {
      title: 'Humanoid Locomotion',
      description: 'Learn the principles of bipedal walking, balance control, and dynamic movement for humanoid robots.',
      duration: '4 weeks',
      level: 'Advanced',
      link: '/docs/intro'
    },
    {
      title: 'Human-Robot Interaction',
      description: 'Explore social robotics, natural language interaction, and collaborative behaviors for humanoid robots.',
      duration: '3 weeks',
      level: 'Intermediate',
      link: '/docs/intro'
    }
  ];

  return (
    <Layout
      title={`Course Modules | ${siteConfig.title}`}
      description="Structured learning modules for Physical AI & Humanoid Robotics">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title">Course Modules</h1>
          <p className="hero__subtitle">Structured learning paths to master Physical AI & Humanoid Robotics</p>
        </div>
      </header>
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              {courses.map((course, idx) => (
                <CourseCard key={idx} {...course} />
              ))}
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}