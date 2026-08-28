import {useEffect} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

export default function CatalogPlaceholder(): React.JSX.Element {
  const catalogUrl = useBaseUrl('/');

  useEffect(() => {
    window.location.replace(catalogUrl);
  }, [catalogUrl]);

  return (
    <main aria-live="polite">
      <p>Opening Models…</p>
    </main>
  );
}
