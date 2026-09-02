import useBaseUrl from '@docusaurus/useBaseUrl';

type Stage = {
  action: [string, string];
  guide: string;
  guideLabel: string;
  number: string;
  success: [string, string];
  title: string;
};

const stages: Stage[] = [
  {
    number: '01',
    title: 'Establish contract',
    action: ['Pin source + datasets', 'Define YAML, recipe, tests'],
    success: ['Complete, reviewable', 'model contract'],
    guide: 'adding-a-model',
    guideLabel: 'Adding a model',
  },
  {
    number: '02',
    title: 'Qualify locally',
    action: ['Validate + fetch inputs', 'Compile + verify variants'],
    success: ['One valid MPK', 'per tested variant'],
    guide: 'local-development',
    guideLabel: 'Local development',
  },
  {
    number: '03',
    title: 'Benchmark MPK',
    action: ['Run the exact MPK', 'on a Modalix DevKit'],
    success: ['Machine-readable', 'performance result'],
    guide: 'benchmarking',
    guideLabel: 'Benchmarking',
  },
  {
    number: '04',
    title: 'Prepare Git',
    action: ['Commit inputs only', 'Document local evidence'],
    success: ['Focused, reviewable', 'pull request'],
    guide: 'contributing',
    guideLabel: 'Contributing',
  },
  {
    number: '05',
    title: 'PR validates',
    action: ['Validate catalog + matrix', 'Keep contributor code off Vulcan'],
    success: ['Safe, deterministic', 'review signal'],
    guide: 'contributing#ci-and-trust-boundaries',
    guideLabel: 'PR checks',
  },
  {
    number: '06',
    title: 'Trusted automation',
    action: ['Compile + measure accuracy', 'Publish + benchmark staging'],
    success: ['Trusted evidence', 'and provenance'],
    guide: 'contributing#after-merge-trusted-staging-automation',
    guideLabel: 'After merge',
  },
];

export default function ContributionPath(): React.JSX.Element {
  const docsBaseUrl = useBaseUrl('/docs');

  return (
    <div className="contribution-path-frame">
      <ol
        className="contribution-path"
        aria-label="Path to a successful model contribution">
        {stages.map((stage, index) => {
          const focalClass = index === stages.length - 1 ? ' contribution-path-stage--focal' : '';

          return (
            <li
              className={`contribution-path-item contribution-path-item--${stage.number}`}
              key={stage.number}>
              <a
                className={`contribution-path-link${focalClass}`}
                href={`${docsBaseUrl}/${stage.guide}`}
                aria-label={`${stage.title}: open ${stage.guideLabel}`}>
                <span className="contribution-path-step" aria-hidden="true">{stage.number}</span>
                <span className="contribution-path-name">{stage.title}</span>
                <span className="contribution-path-copy contribution-path-action">
                  <span>{stage.action[0]}</span>
                  <span>{stage.action[1]}</span>
                </span>
                <span className="contribution-path-rule" aria-hidden="true" />
                <span className="contribution-path-label">Success</span>
                <span className="contribution-path-copy contribution-path-result">
                  <span>{stage.success[0]}</span>
                  <span>{stage.success[1]}</span>
                </span>
                <span className="contribution-path-guide">{stage.guideLabel} →</span>
              </a>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
