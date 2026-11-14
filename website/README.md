# Unified Model Browser portal

The combined static site serves the Model Browser catalog at `/` and the
Docusaurus developer guide at `/docs`. Docusaurus consumes the repository's
top-level `docs/` directory; the assembly step adds the client-only app from
`artifact-browser/` and fingerprints its mutable assets.

Build and serve the combined portal with Node.js 20 or newer:

```bash
npm ci
npm run typecheck
npm run build:portal
npm run serve:portal
```

Open <http://localhost:4173> for the catalog and
<http://localhost:4173/docs> for documentation. Deployment workflows provide
their own origin and base path: `/models/` for the Developer Center and the
repository-assigned path for GitHub Pages.
