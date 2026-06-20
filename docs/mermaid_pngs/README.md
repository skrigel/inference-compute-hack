# Rendered Mermaid PNGs

Generated: 2026-06-20

This folder contains PNG renders of every Mermaid code block currently tracked
in the repo's Markdown files.

| PNG | Source Mermaid block | Notes |
|---|---|---|
| `readme-1.png` | `README.md:56` | Architecture overview from the top-level README. |
| `plan-1.png` | `PLAN.md:98` | Architecture and ownership layout from the project plan. Rendered from a normalized temporary Mermaid source because the current Mermaid CLI requires quoted node labels containing parentheses. The source Markdown was not changed. |

Render command used:

```bash
PUPPETEER_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  npx -y @mermaid-js/mermaid-cli@latest -i <source.mmd> -o <target.png> -b white
```
