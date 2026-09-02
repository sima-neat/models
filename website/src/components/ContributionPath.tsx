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
      <svg
        className="contribution-path"
        viewBox="0 0 1344 368"
        role="img"
        aria-labelledby="contribution-path-title contribution-path-desc">
        <title id="contribution-path-title">Path to a successful model contribution</title>
        <desc id="contribution-path-desc">
          Six linked stages establish the model contract, qualify it locally,
          benchmark the exact MPK, prepare the Git contribution, validate the
          pull request safely, and run trusted staging automation after merge.
          Each stage links to its detailed guide.
        </desc>
        <defs>
          <marker id="contribution-path-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path className="contribution-path-arrowhead" d="M 0 0 L 8 4 L 0 8 Z" />
          </marker>
        </defs>

        <g className="contribution-path-connectors" aria-hidden="true">
          <line x1="216" y1="184" x2="236" y2="184" markerEnd="url(#contribution-path-arrow)" />
          <line x1="440" y1="184" x2="460" y2="184" markerEnd="url(#contribution-path-arrow)" />
          <line x1="664" y1="184" x2="684" y2="184" markerEnd="url(#contribution-path-arrow)" />
          <line x1="888" y1="184" x2="908" y2="184" markerEnd="url(#contribution-path-arrow)" />
          <line x1="1112" y1="184" x2="1132" y2="184" markerEnd="url(#contribution-path-arrow)" />
        </g>

        {stages.map((stage, index) => {
          const x = 16 + index * 224;
          const center = x + 100;
          const focalClass = index === stages.length - 1 ? ' contribution-path-stage--focal' : '';

          return (
            <a
              className={`contribution-path-link${focalClass}`}
              href={`${docsBaseUrl}/${stage.guide}`}
              aria-label={`${stage.title}: open ${stage.guideLabel}`}
              key={stage.number}>
              <rect className="contribution-path-stage" x={x} y="40" width="200" height="288" rx="4" />
              <text className="contribution-path-step" x={center} y="72" textAnchor="middle">
                {stage.number}
              </text>
              <text className="contribution-path-name" x={center} y="108" textAnchor="middle">
                {stage.title}
              </text>
              <text className="contribution-path-copy" x={center} y="148" textAnchor="middle">
                <tspan x={center} dy="0">{stage.action[0]}</tspan>
                <tspan x={center} dy="20">{stage.action[1]}</tspan>
              </text>
              <line className="contribution-path-rule" x1={x + 16} y1="196" x2={x + 184} y2="196" />
              <text className="contribution-path-label" x={center} y="224" textAnchor="middle">
                SUCCESS
              </text>
              <text className="contribution-path-copy" x={center} y="248" textAnchor="middle">
                <tspan x={center} dy="0">{stage.success[0]}</tspan>
                <tspan x={center} dy="20">{stage.success[1]}</tspan>
              </text>
              <text className="contribution-path-guide" x={center} y="308" textAnchor="middle">
                {stage.guideLabel} →
              </text>
            </a>
          );
        })}
      </svg>
    </div>
  );
}
