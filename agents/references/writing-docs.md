# Writing docs pages

The page standard for `docs/`, the published site. Read it before you write or change a
page there. `tests/test_docs.py` checks the parts marked (tested).

## Who reads docs

Two readers, both outside the team's daily context:

- someone who wants to reproduce a published model on their own machine and DevKit
- someone who adds or updates a model

Neither reader has seen the CI internals. What CI, publication, and operations do belongs
in `agents/references/`. A docs page states only the outcome the reader sees, such as "CI
publishes the result after the merge".

A docs page links only to other docs pages, to files a reader opens (a schema, a script,
a model folder), and to external sites. It never links to `AGENTS.md`, `agents/`, or
`design-doc/` (tested).

## Two page kinds

Every page is one kind. Reference tables that a task page needs go at its end.

A task page tells the reader how to do one job:

```text
# Verb-first title
Opening: two sentences. What the reader has at the end, and when they need this page.
## Before you start     what must exist, each item with a command that checks it
## Steps                numbered, in the order the reader performs them
## If something fails   the error as printed, its cause, the fix
## Next                 the one page that follows
```

A reference page lets the reader look one thing up. It opens with what it describes and
when to look there. Items follow in the order the reader meets them, for example the
field order of `model.yaml`. Each item gives its meaning, allowed values, default, and
one example.

Every page starts with an H1 and at least one sentence before its first H2 (tested).
Headings stop at H3.

## Steps

- One action per step.
- Show the command in a fenced block with its language (tested). Use values from one
  real model, and keep that model for the whole page.
- After the command, say what the reader should see, so they know it worked.
- Put the reason next to the step it explains, in one sentence starting from
  "because" or "so that". Longer background goes to `docs/concepts.md` behind a link.
- Run every command before the page merges, in the venue it names. A page whose
  commands were not run says so in its pull request.

## Language

- Define a term in `docs/concepts.md` and at its first use on the page. Then keep that
  name. The names are the ones in the `## Vocabulary` section of `AGENTS.md`.
- Keep instructions within about 20 words and descriptions within about 25.
- Use the active voice and name who acts: "CI publishes the report".
- Use the plain word: "use", "start", "because".
- Write no em dashes and no semicolons. Use sentence-case headings.
- Short means deleting sentences that change nothing the reader does. Keep every value,
  condition, exception, and failure.
- State each fact on one page. Other pages link to it.
- Describe what the source does today. Leave out dated status, issue numbers, account
  identifiers, and internal host names a reader outside the team cannot reach.

## Before a page merges

1. Check every command, flag, file name, and value against current source.
2. Give the page alone to a reader with no other context and have them follow it for
   one real model. Fix each place where they had to guess.
3. Add the page to `website/sidebars.ts` and to the index in `AGENTS.md` (tested).
4. Run `python -m pytest tests/test_docs.py` and `npm run build:portal --prefix website`.
