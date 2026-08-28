import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const steps = [
  {
    number: '01',
    title: 'Describe',
    text: 'Pin immutable source bytes, ownership, licensing, taxonomy, tensors, and the model route.',
    link: '/docs/adding-a-model',
  },
  {
    number: '02',
    title: 'Verify',
    text: 'Validate the catalog, fetch checksummed inputs, and inspect the exact compilation matrix.',
    link: '/docs/local-development',
  },
  {
    number: '03',
    title: 'Qualify',
    text: 'Move through trusted compilation and Modalix evidence without blurring the security boundary.',
    link: '/docs/contributing',
  },
];

function Home(): React.JSX.Element {
  return (
    <Layout
      title="Developer documentation"
      description="Introduce, validate, and qualify models for the SiMa Neat Model Registry.">
      <main>
        <section className={styles.hero}>
          <div className={styles.heroGrid}>
            <div className={styles.heroCopy}>
              <p className={styles.eyebrow}>Model Registry / Developer documentation</p>
              <Heading as="h1">A model is more than an artifact.</Heading>
              <p className={styles.lede}>
                Ship the source, contract, recipe, and evidence as one reproducible contribution.
              </p>
              <div className={styles.actions}>
                <Link className={clsx('button', styles.primaryAction)} to="/docs/adding-a-model">
                  Add your first model
                </Link>
                <Link className={styles.secondaryAction} to="/docs/metadata-schema">
                  Read the YAML contract <span aria-hidden="true">→</span>
                </Link>
              </div>
            </div>

            <aside className={styles.contract} aria-label="Model contribution contract">
              <div className={styles.contractHeader}>
                <span>model.yaml</span>
                <span className={styles.contractState}>schema v2</span>
              </div>
              <pre>
                <code>{`source: immutable\ninterface: explicit\nrecipe: deterministic\nevidence: reviewable`}</code>
              </pre>
              <p>Local validation proves the declaration. Trusted infrastructure proves the artifact.</p>
            </aside>
          </div>
        </section>

        <section className={styles.journey} aria-labelledby="journey-heading">
          <div className={styles.sectionIntro}>
            <p className={styles.eyebrow}>Contribution path</p>
            <Heading as="h2" id="journey-heading">Three boundaries. One audit trail.</Heading>
          </div>
          <ol className={styles.stepList}>
            {steps.map((step) => (
              <li key={step.number}>
                <Link to={step.link} className={styles.stepLink}>
                  <span className={styles.stepNumber}>{step.number}</span>
                  <span className={styles.stepBody}>
                    <strong>{step.title}</strong>
                    <span>{step.text}</span>
                  </span>
                  <span className={styles.stepArrow} aria-hidden="true">↗</span>
                </Link>
              </li>
            ))}
          </ol>
        </section>

        <section className={styles.referenceBand} aria-labelledby="reference-heading">
          <div>
            <p className={styles.eyebrow}>Keep close at hand</p>
            <Heading as="h2" id="reference-heading">The contracts behind every build.</Heading>
          </div>
          <nav className={styles.referenceLinks} aria-label="Model Registry references">
            <Link to="/docs/metadata-schema">model.yaml reference</Link>
            <Link to="/docs/model-taxonomy">Taxonomy vocabulary</Link>
            <Link to="/docs/benchmarking">Benchmarking guide</Link>
            <Link to="/docs/architecture">System architecture</Link>
          </nav>
        </section>
      </main>
    </Layout>
  );
}

export default Home;
