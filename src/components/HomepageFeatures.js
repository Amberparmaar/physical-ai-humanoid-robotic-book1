import React from "react";
import clsx from "clsx";
import styles from "./HomepageFeatures.module.css";

const FeatureList = [
  {
    title: "Physical AI & Humanoid Robotics",
    imageUrl: "/img/ai.png",
    description: (
      <>
        Learn about the cutting-edge field of Physical AI, where artificial
        intelligence systems interact directly with the physical world through
        robotic platforms. Understand how humanoid robots represent one of the
        most ambitious frontiers in robotics with their human-like form and
        capabilities.
      </>
    ),
  },
  {
    title: "AI-Native Learning Experience",
    imageUrl: "/img/robot.png",
    description: (
      <>
        Experience a revolutionary approach to learning with our AI-native
        textbook. Leverage Retrieval-Augmented Generation (RAG) systems for
        real-time, context-aware responses to your questions, with personalized
        content adaptation based on your learning style and progress.
      </>
    ),
  },
  {
    title: "Integrated Technology Stack",
    imageUrl: "/img/tech-stack.jpg",
    description: (
      <>
        Master the four core technologies of modern robotics: ROS2 for
        communication, Gazebo for simulation, NVIDIA Isaac for AI acceleration,
        and VLA models for vision-language-action integration in humanoid
        applications.
      </>
    ),
  },
];

function Feature({ imageUrl, title, description }) {
  return (
    <div className={clsx("col col--4")}>
      <div className={styles.featureCard}>
        <div className="text--center">
          {imageUrl ? (
            <img src={imageUrl} alt={title} className={styles.featureImage} />
          ) : null}
        </div>
        <div className="text--center padding-horiz--md">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
