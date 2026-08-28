import useIsBrowser from '@docusaurus/useIsBrowser';
import type {Props} from '@theme/ColorModeToggle';

export default function ColorModeToggle({className, buttonClassName, onChange}: Props): React.JSX.Element {
  const isBrowser = useIsBrowser();
  const dark = isBrowser && document.documentElement.dataset.theme === 'dark';
  const nextMode = dark ? 'light' : 'dark';

  return (
    <div className={className}>
      <button
        className={`clean-btn portal-theme-toggle ${buttonClassName ?? ''}`}
        type="button"
        onClick={() => onChange(nextMode)}
        disabled={!isBrowser}
        aria-label={`Switch to ${nextMode} mode`}
        title={`Switch to ${nextMode} mode`}>
        <svg className="portal-theme-icon portal-theme-icon--sun" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="3.5" />
          <path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
        </svg>
        <svg className="portal-theme-icon portal-theme-icon--moon" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M20.3 15.2A8.5 8.5 0 0 1 8.8 3.7 8.5 8.5 0 1 0 20.3 15.2Z" />
        </svg>
      </button>
    </div>
  );
}
