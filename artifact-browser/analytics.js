// Consent-gated GA4 for the model browser and its docs.
//
// The Developer Center keeps one cookie decision for the whole developer.sima.ai origin, written by the
// docs banner in sima-neat/core. This module reads and writes the same localStorage entry, so a visitor who
// already chose on the docs is not asked again, and shows the same banner when no decision exists yet.
// gtag loads only after analytics is accepted. Nothing runs without a measurement ID or off the site's host.

export const CONSENT_KEY = "sima-developer-center-cookie-consent";
export const CONSENT_VERSION = 1;
const CONSENT_EVENT = "neat:analytics-consent";

const deniedConsent = {
  analytics_storage: "denied",
  ad_storage: "denied",
  ad_user_data: "denied",
  ad_personalization: "denied",
};
const grantedConsent = { ...deniedConsent, analytics_storage: "granted" };

let settings = null;
let gtagLoaded = false;
let lastTrackedPath = "";

export function analyticsEnabled(analytics, hostname) {
  return Boolean(analytics?.measurementId && analytics.host && analytics.host === hostname);
}

export function parseConsent(value) {
  try {
    const parsed = value ? JSON.parse(value) : null;
    return parsed && parsed.version === CONSENT_VERSION ? parsed : null;
  } catch {
    return null;
  }
}

function storedConsent() {
  try {
    return parseConsent(window.localStorage.getItem(CONSENT_KEY));
  } catch {
    return null; // Storage can be unavailable in strict privacy modes.
  }
}

function storeConsent(analytics) {
  const next = { version: CONSENT_VERSION, analytics, marketing: false, updatedAt: new Date().toISOString() };
  try {
    window.localStorage.setItem(CONSENT_KEY, JSON.stringify(next));
  } catch {
    // The choice still applies to this page view.
  }
  window.dispatchEvent(new CustomEvent(CONSENT_EVENT, { detail: next }));
  return next;
}

function ensureGtag() {
  window.dataLayer = window.dataLayer || [];
  if (!window.gtag) {
    window.gtag = function gtag() {
      window.dataLayer.push(arguments);
    };
    window.gtag("consent", "default", deniedConsent);
  }
}

function currentPath() {
  return `${window.location.pathname}${window.location.search}`;
}

function trackPageView() {
  if (!gtagLoaded || !storedConsent()?.analytics) return;
  const path = currentPath();
  if (path === lastTrackedPath) return;
  lastTrackedPath = path;
  window.gtag("event", "page_view", {
    page_title: document.title,
    page_location: window.location.href,
    page_path: path,
  });
}

function loadGtag() {
  if (gtagLoaded) return;
  gtagLoaded = true;
  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(settings.measurementId)}`;
  document.head.append(script);
  window.gtag("js", new Date());
  window.gtag("config", settings.measurementId, { anonymize_ip: true, send_page_view: false });
}

function applyConsent(consent) {
  ensureGtag();
  window.gtag("consent", "update", consent.analytics ? grantedConsent : deniedConsent);
  if (consent.analytics) {
    loadGtag();
    // The docs set their page title just after the first route update.
    window.setTimeout(trackPageView, 80);
  }
}

// Page views follow pushState and Back/Forward only. replaceState rewrites the URL in place (filter state,
// canonical redirects) and is not a new page. The short delay lets the page set its new title first.
function watchNavigation() {
  const schedule = () => window.setTimeout(trackPageView, 80);
  const pushState = window.history.pushState;
  window.history.pushState = function trackedPushState(...args) {
    const result = pushState.apply(this, args);
    schedule();
    return result;
  };
  window.addEventListener("popstate", schedule);
}

const STYLE_ID = "cookie-consent-style";
const STYLES = `
.cookie-consent { position: fixed; z-index: 2147483000; left: max(1rem, env(safe-area-inset-left)); right: max(1rem, env(safe-area-inset-right)); bottom: max(1rem, env(safe-area-inset-bottom)); width: min(640px, calc(100vw - 2rem)); max-height: calc(100vh - 2rem); overflow: auto; margin: 0 auto; display: flex; flex-direction: column; gap: 1.1rem; padding: 1.1rem; box-sizing: border-box; border: 1px solid rgba(29, 95, 115, 0.24); border-radius: 8px; color: var(--ink, var(--ifm-font-color-base, #0b1220)); font-family: var(--font-body, var(--ifm-font-family-base, Roboto, sans-serif)); background: linear-gradient(135deg, rgba(29, 95, 115, 0.08), rgba(47, 160, 82, 0.08)), var(--surface-raised, var(--ifm-background-surface-color, #ffffff)); box-shadow: 0 18px 50px rgba(15, 23, 42, 0.2); }
.cookie-consent__eyebrow { margin: 0 0 0.2rem; color: var(--sima-blue, #5998dd); font-size: 0.74rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
.cookie-consent__title { margin: 0; font-size: 1.1rem; line-height: 1.25; }
.cookie-consent__text, .cookie-consent__note { margin: 0.35rem 0 0; color: var(--ink-soft, var(--ifm-color-emphasis-700, #5f6972)); font-size: 0.9rem; line-height: 1.45; }
.cookie-consent__note { font-size: 0.78rem; }
.cookie-consent__details { margin-top: 0.85rem; padding-top: 0.75rem; border-top: 1px solid var(--line, var(--ifm-color-emphasis-300, #dfe3e6)); }
.cookie-consent__details h3 { margin: 0.45rem 0 0.15rem; font-size: 0.84rem; }
.cookie-consent__details p { margin: 0; color: var(--ink-soft, var(--ifm-color-emphasis-700, #5f6972)); font-size: 0.8rem; line-height: 1.45; }
.cookie-consent__actions { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: auto; padding-top: 0.2rem; }
.cookie-consent__actions button { flex: 1 1 220px; min-height: 40px; padding: 9px 16px; border: 1px solid var(--line-strong, var(--ifm-color-emphasis-400, #c7cdd2)); border-radius: 8px; font: inherit; font-size: 14px; font-weight: 600; white-space: nowrap; cursor: pointer; color: var(--ink, var(--ifm-font-color-base, #0b1220)); background: transparent; }
.cookie-consent__actions .cookie-consent__primary { border-color: #5a9bdc; color: #f8fbff; background: #5a9bdc; }
.cookie-consent__actions .cookie-consent__primary:hover, .cookie-consent__actions .cookie-consent__primary:focus-visible { border-color: #4386c6; background: #4386c6; }
.cookie-consent__toggle { display: flex; gap: 1rem; align-items: center; justify-content: space-between; margin-top: 0.85rem; padding: 0.75rem 0.85rem; border: 1px solid var(--line, var(--ifm-color-emphasis-300, #dfe3e6)); border-radius: 8px; background: rgba(255, 255, 255, 0.65); }
.cookie-consent__toggle strong, .cookie-consent__toggle small { display: block; }
.cookie-consent__toggle small { margin-top: 0.15rem; color: var(--ink-soft, var(--ifm-color-emphasis-700, #5f6972)); }
.cookie-consent__toggle input { width: 1.15rem; height: 1.15rem; accent-color: #5a9bdc; }
html[data-theme="dark"] .cookie-consent { border-color: rgba(113, 193, 212, 0.32); background: linear-gradient(135deg, rgba(113, 193, 212, 0.12), rgba(47, 160, 82, 0.1)), var(--surface-raised, var(--ifm-background-surface-color, #252a2e)); box-shadow: 0 18px 50px rgba(0, 0, 0, 0.55); }
html[data-theme="dark"] .cookie-consent__toggle { background: rgba(255, 255, 255, 0.04); }
@media (max-width: 600px) { .cookie-consent { width: auto; } .cookie-consent__actions button { flex-basis: 100%; } }
`;

const DETAILS = `
  <div class="cookie-consent__details">
    <h3>What is collected</h3>
    <p>When analytics is accepted, the site may collect aggregate usage such as page views, navigation paths, and engagement events. The site should not send names, email addresses, account identifiers, or other personal content to analytics.</p>
    <h3>Your choices</h3>
    <p>You can accept or reject optional analytics cookies. The site works either way, and you can reopen these settings from Cookie preferences.</p>
  </div>`;

function installStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = STYLES;
  document.head.append(style);
}

function closeConsentUi() {
  document.querySelectorAll(".cookie-consent").forEach((element) => element.remove());
}

function showConsentUi({ preferences }) {
  closeConsentUi();
  installStyles();
  const analyticsOn = storedConsent()?.analytics ?? false;
  const section = document.createElement("section");
  section.className = `cookie-consent cookie-consent--${preferences ? "panel" : "banner"}`;
  section.setAttribute("aria-label", preferences ? "Cookie preferences" : "Cookie notice");
  section.innerHTML = preferences ? `
    <div class="cookie-consent__content">
      <p class="cookie-consent__eyebrow">Privacy preferences</p>
      <h2 class="cookie-consent__title">Cookie settings</h2>
      <p class="cookie-consent__text">Necessary cookies keep the site working. Analytics cookies help SiMa.ai understand aggregate usage. Marketing cookies are not used on this site today.</p>
      <label class="cookie-consent__toggle">
        <span><strong>Analytics</strong><small>Google Analytics 4, loaded only after consent.</small></span>
        <input type="checkbox" ${analyticsOn ? "checked" : ""} />
      </label>${DETAILS}
      <p class="cookie-consent__note">Global Privacy Control is respected by keeping marketing disabled.</p>
    </div>
    <div class="cookie-consent__actions">
      <button type="button" class="cookie-consent__primary" data-cookie-choice="save">Save settings</button>
      <button type="button" data-cookie-choice="reject">Reject optional cookies</button>
    </div>` : `
    <div class="cookie-consent__content">
      <p class="cookie-consent__eyebrow">Privacy</p>
      <h2 class="cookie-consent__title">Help improve the Developer Center</h2>
      <p class="cookie-consent__text">We use optional Google Analytics cookies to understand aggregate usage. The site works without them.</p>${DETAILS}
    </div>
    <div class="cookie-consent__actions">
      <button type="button" class="cookie-consent__primary" data-cookie-choice="accept">Accept analytics</button>
      <button type="button" data-cookie-choice="reject">Reject optional cookies</button>
    </div>`;
  section.addEventListener("click", (event) => {
    const choice = event.target.closest?.("[data-cookie-choice]")?.dataset.cookieChoice;
    if (!choice) return;
    const accepted = choice === "accept" || (choice === "save" && section.querySelector("input[type='checkbox']").checked);
    applyConsent(storeConsent(accepted));
    closeConsentUi();
  });
  document.body.append(section);
}

// Any element with data-cookie-preferences reopens the settings, including ones rendered after start.
function bindPreferenceLinks() {
  document.addEventListener("click", (event) => {
    if (!event.target.closest?.("[data-cookie-preferences]")) return;
    event.preventDefault();
    showConsentUi({ preferences: true });
  });
}

export function startAnalytics(analytics) {
  if (settings || typeof window === "undefined") return;
  if (!analyticsEnabled(analytics, window.location.hostname)) return;
  settings = analytics;
  document.documentElement.classList.add("analytics-consent-enabled");
  ensureGtag();
  watchNavigation();
  bindPreferenceLinks();
  const consent = storedConsent();
  if (consent) applyConsent(consent);
  else showConsentUi({ preferences: false });
}
