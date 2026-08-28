# Developer documentation website

The Docusaurus site consumes the repository's top-level `docs/` directory.
Run it from this directory with Node.js 20 or newer:

```bash
npm ci
npm start
```

Build the GitHub Pages artifact with:

```bash
npm run typecheck
npm run build
```

The production workflow obtains the assigned origin and base path from GitHub
Pages before building. Local builds default to the repository's private Pages
origin at `/` and can override those values with `DOCS_SITE_URL` and
`DOCS_BASE_URL`.
