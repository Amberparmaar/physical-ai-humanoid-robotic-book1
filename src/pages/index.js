import React from "react";
import clsx from "clsx";
import Link from "@docusaurus/Link";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import HomepageFeatures from "@site/src/components/HomepageFeatures";

import styles from "./index.module.css";

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx("hero hero--primary", styles.heroBanner)}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <h1 className="hero__title">{siteConfig.title}</h1>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <p>
              Discover the future of robotics where artificial intelligence
              meets physical interaction. This comprehensive textbook covers all
              aspects of Physical AI and Humanoid Robotics.
            </p>
            <div className={styles.buttons}>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro"
              >
                Start Learning - 5 min ⏱️
              </Link>
            </div>
          </div>
          <div className="col col--6 text--center">
            <img
              src="img/robot.png"
              alt="Robot Illustration"
              className={styles.heroImage}
              style={{ maxWidth: "100%", height: "auto" }}
            />
          </div>
        </div>
      </div>
    </header>
  );
}

function StatsSection() {
  return (
    <section className={styles.stats}>
      <div className="container padding-vert--md">
        <div className="row">
          <div className="col col--3 text--center">
            <h2 className={styles.statNumber}>10+</h2>
            <p className={styles.statLabel}>Key Technologies</p>
          </div>
          <div className="col col--3 text--center">
            <h2 className={styles.statNumber}>20+</h2>
            <p className={styles.statLabel}>Learning Modules</p>
          </div>
          <div className="col col--3 text--center">
            <h2 className={styles.statNumber}>50+</h2>
            <p className={styles.statLabel}>Concepts Explained</p>
          </div>
          <div className="col col--3 text--center">
            <h2 className={styles.statNumber}>100%</h2>
            <p className={styles.statLabel}>AI-Enhanced</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className={styles.cta}>
      <div className="container padding-vert--lg text--center">
        <h2>Ready to Dive into Physical AI & Humanoid Robotics?</h2>
        <p>
          Join thousands of students and professionals learning with our
          AI-native textbook.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg margin-right--lg"
            to="/docs/intro"
          >
            Get Started Now
          </Link>
          <Link className="button button--secondary button--lg" to="/courses">
            Browse Course Modules
          </Link>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="AI-native textbook on Physical AI & Humanoid Robotics"
    >
      <HomepageHeader />
      <main>
        <StatsSection />
        <HomepageFeatures />
        <CTASection />
      </main>
    </Layout>
  );
}
