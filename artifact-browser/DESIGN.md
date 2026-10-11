# Model browser design

## Users

Model engineers and application developers use the Model Registry to find a
suitable verified model quickly, compare completed builds, inspect provenance,
and download deployable artifacts. Search and model compatibility are the
primary browsing tasks; operational publication state is not part of the
public discovery experience.

## Brand Personality

Calm, precise, and technical. The interface should create confidence through
clear hierarchy, restrained visual language, and trustworthy metadata rather
than decoration or dashboard density.

## Aesthetic Direction

Refined minimalism with generous space and progressive disclosure. Light and
dark themes remain supported. Search is the dominant action; advanced facets
stay collapsed until requested. Avoid crowded toolbars, repeated status
indicators, and decorative effects that compete with model information.

The catalog is a grid of model cards because people recognize a model by what
it does before they know its name. Each card carries one thing per role: the
model's hero illustration, its name, a one-line description, and its tasks. A
card states what the model is, not how a build measured; numbers, precision,
and build state belong on the model page, where they have context. Card
borders and fills mark the cards as the objects to choose; the model page
itself uses sections and rules instead of nested cards.

## Design Principles

1. Make finding a model the single primary action.
2. Show only completed, usable builds in public discovery.
3. Hide advanced filtering until it is useful, while making active filters
   obvious and easy to clear.
4. Search across names and sanctioned category labels using plain language.
5. Preserve accessible labels, keyboard operation, responsive behavior, and
   reduced-motion support while simplifying the interface.
