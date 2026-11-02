// Hash-addressed public derivatives; absent configuration keeps the registry route.
export function heroImageSources(documentation, apiBase, cdnBase = "") {
  const hero = documentation?.hero;
  if (!hero?.artifact_name || !/^[0-9a-f]{64}$/.test(hero.artifact_sha256 || "") || !documentation?.run_id) return null;
  const fallback = `${apiBase}/v1/runs/${encodeURIComponent(documentation.run_id)}/artifacts/${encodeURIComponent(hero.artifact_name)}/content`;
  return { src: cdnBase ? `${cdnBase.replace(/\/$/, "")}/${hero.artifact_sha256}.png` : fallback, fallback };
}

// Image errors do not bubble. Capture them before the first catalog render.
// Two CDN retries, then two API retries; no cache-busting immutable URLs.
export function installHeroImageRecovery(root, schedule = setTimeout) {
  const attempts = new WeakMap();
  root.addEventListener("error", (event) => {
    const image = event.target;
    if (image?.tagName !== "IMG") return;
    const fallback = image.getAttribute("data-hero-fallback");
    if (!fallback) return;
    const attempt = attempts.get(image) || 0;
    const onFallback = image.getAttribute("src") === fallback;
    const limit = onFallback ? 4 : 2;
    if (attempt >= limit && onFallback) return;
    attempts.set(image, attempt + 1);
    schedule(() => {
      if (!image.isConnected) return;
      if (attempt >= 2) image.setAttribute("src", fallback);
      else image.setAttribute("src", image.getAttribute("src"));
    }, [500, 1500, 3000, 5000][attempt]);
  }, true);
}
