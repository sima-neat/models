#!/usr/bin/env node
// Usage (actions/github-script): await require('./scripts/authorize-model-browser-deployment.cjs')({github, context, core});
// Verify the protected environment and dispatching owner's approval before any deployment credentials are issued.
module.exports = async function authorizeModelBrowserDeployment({github, context, core}) {
  if (!['staging', 'production'].includes(process.env.DEPLOY_ENV) ||
      !['developer', 'customer'].includes(process.env.DEPLOY_AUDIENCE)) {
    throw new Error('Unsupported deployment environment or audience.');
  }
  const {data} = await github.rest.repos.getEnvironment({
    ...context.repo, environment_name: 'model-browser-deploy'
  });
  const gate = data.protection_rules?.find(rule => rule.type === 'required_reviewers');
  const reviewers = gate?.reviewers || [];
  if (data.can_admins_bypass !== false || reviewers.length !== 1 ||
      reviewers[0].type !== 'Team' || reviewers[0].reviewer?.id !== 20041032) {
    throw new Error('Deployment requires the owners-models approval gate with admin bypass disabled.');
  }
  if (Number(process.env.GITHUB_RUN_ATTEMPT) !== 1 || context.actor !== process.env.GITHUB_TRIGGERING_ACTOR) {
    throw new Error('Dispatch a new run; reruns cannot reuse an earlier owner approval.');
  }
  const {data: approvals} = await github.request('GET /repos/{owner}/{repo}/actions/runs/{run_id}/approvals', {
    ...context.repo, run_id: context.runId
  });
  const ownApproval = approvals.some(review => review.state === 'approved' &&
    review.user?.login === context.actor &&
    review.environments?.some(environment => environment.name === 'model-browser-deploy'));
  if (!ownApproval) {
    throw new Error('Only the dispatching models owner may approve this deployment. Ask an owner to dispatch a new run.');
  }
  core.notice(`Owner-approved deployment of ${context.sha}; destination ${process.env.DEPLOY_ENV}.`);
};
