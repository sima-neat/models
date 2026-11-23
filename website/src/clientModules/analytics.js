import siteConfig from '@generated/docusaurus.config';
import {startAnalytics} from '../../../artifact-browser/analytics.js';

// Same consent banner and GA4 stream as the model browser, configured once in scripts/portal-config.mjs.
export function onRouteDidUpdate() {
  startAnalytics(siteConfig.customFields?.analytics);
}
