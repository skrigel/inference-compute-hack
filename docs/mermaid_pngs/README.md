# Rendered Mermaid Diagrams

Generated: 2026-06-20

This folder contains high-quality renders of every Mermaid code block currently
tracked in the repo's Markdown files. Prefer the `.svg` files for slides, decks,
and docs because they are vector renders. The `.png` files are 2x-scale fallbacks
for tools that cannot import SVG.

| Render | Source Mermaid block | Notes |
|---|---|---|
| `readme-1.svg`, `readme-1.png` | `README.md:56` | Architecture overview from the top-level README. PNG fallback is `2454 x 622`. |
| `plan-1.svg`, `plan-1.png` | `PLAN.md:98` | Architecture and ownership layout from the project plan. PNG fallback is `3168 x 1348`. Rendered from a normalized temporary Mermaid source because the current Mermaid CLI requires quoted node labels containing parentheses. The source Markdown was not changed. |

SVG render command:

```bash
PUPPETEER_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  npx -y @mermaid-js/mermaid-cli@latest -i <source.mmd> -o <target.svg> -b white -t neutral
```

High-resolution PNG fallback command:

```bash
PUPPETEER_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  npx -y @mermaid-js/mermaid-cli@latest -i <source.mmd> -o <target.png> -b white -t neutral -w 1600 -H 900 -s 2
```
