# Frontend UX Improvements Design

**Date:** 2026-06-20
**Status:** Approved

## Overview

Four improvements to the frontend to support real backend testing and better UX:

1. Settings toggle for mock/live API mode
2. Built-in corpora (demo + browsecomp) on homepage
3. Remove auto-execute query behavior
4. Document preview on result click

## 1. Settings Toggle for Mock/Live Mode

### Problem
Frontend defaults to mock mode. Testing live backend requires setting `VITE_DATA_MODE=live` env var and restarting the dev server.

### Solution
Add a Settings page with a toggle to switch between mock and live mode at runtime.

### Implementation

**New files:**
- `src/hooks/useSettings.ts` - Hook for reading/writing settings to localStorage
- `src/pages/SettingsPage.tsx` - Settings page with API mode toggle

**Modified files:**
- `src/lib/api.ts` - Read mode from localStorage instead of env var; support runtime mode switching
- `src/App.tsx` - Add `/settings` route
- `src/App.tsx` or header component - Add settings link (gear icon) in top-right area

**Behavior:**
- Settings stored in `localStorage` under key `"api-mode"`
- Values: `"mock"` (default) | `"live"`
- Toggle label: "Use live backend"
- Description: "Connect to real backend at localhost:8000"
- Mode change takes effect immediately (React state triggers API re-initialization)

## 2. Built-in Corpora on Homepage

### Problem
Homepage has "Try Demo Corpus" button but no easy access to browsecomp. Users must know to navigate to `/search/browsecomp` directly.

### Solution
Add a "Built-in Corpora" section on the homepage showing both demo and browsecomp as cards.

### Implementation

**Modified files:**
- `src/pages/HomePage.tsx` - Add Built-in Corpora section

**UI:**
- Section title: "Built-in Corpora"
- Two cards side by side:

| Card | Title | Description | Action |
|------|-------|-------------|--------|
| Demo | Demo Corpus | 7 code snippets for quick testing | → `/search/demo` |
| BrowseComp | BrowseComp+ | 1,000 web documents | → `/search/browsecomp` |

- Cards include: name, doc count badge, description, "Open" button
- Remove duplicate "Try Demo Corpus" from Quick Start section

## 3. Remove Auto-Execute Query

### Problem
When navigating to `/search/:corpusId`, the page auto-runs a seed query ("every place we retry a network call without backoff"). This wastes compute and confuses users who haven't entered anything yet.

### Solution
Load corpus (ingest) on mount but don't auto-execute any query. Show empty state until user submits.

### Implementation

**Modified files:**
- `src/hooks/useDashboard.ts` - Remove `seedQuery` parameter and auto-run logic (lines 278-285)
- `src/pages/SearchPage.tsx` - Remove `DEFAULT_QUERY` constant; change input to empty with placeholder

**Behavior:**
- Corpus ingests on mount (existing behavior, keep it)
- Query input starts empty with placeholder: "Type a query..."
- Results area shows: "Enter a query and click Scan to search"
- Query only runs when user clicks Scan or presses Enter

## 4. Document Preview

### Problem
Clicking result rows only has +/- refinement buttons. No way to see document details or preview content.

### Solution
Two-step preview: click row shows metadata card; card has button to open full document preview.

### Implementation

**New files:**
- `src/components/DocumentPreview.tsx` - Metadata card and full preview panel components

**Modified files:**
- `src/pages/SearchPage.tsx` - Add click handler on result rows (excluding +/- buttons)

**Metadata Card (Modal/Slide-over):**
- Title (header)
- Type badge
- Category
- Year
- Path (if available)
- Repo (if available)
- Score with visual bar
- "Preview Document" button
- Close button (X)

**Full Document Panel:**
- Opens when clicking "Preview Document"
- Shows metadata header + full document text
- Scrollable content area
- Close button

**Interaction:**
- Click result row → metadata card opens
- Click +/- buttons → refinement (existing behavior, unchanged)
- Click outside card or X → card closes
- Click "Preview Document" → full panel opens

## Architecture

### State Management
All new state uses local React state + localStorage. No new global stores.

```
localStorage
├── api-mode: "mock" | "live"
└── (future settings can go here)

Component State
├── SettingsPage: reads/writes api-mode
├── SearchPage: selectedDocument (for preview)
└── DocumentPreview: isFullPreviewOpen
```

### File Structure
```
src/
├── hooks/
│   └── useSettings.ts (new)
├── pages/
│   ├── HomePage.tsx (modified)
│   ├── SearchPage.tsx (modified)
│   └── SettingsPage.tsx (new)
├── components/
│   └── DocumentPreview.tsx (new)
└── lib/
    └── api.ts (modified)
```

## Testing

- Settings toggle: verify localStorage persists, mode switches without reload
- Built-in corpora: verify both cards navigate correctly
- No auto-query: verify corpus loads but no results until user submits
- Document preview: verify click opens card, preview button opens full panel, close works

## Out of Scope

- Settings sync across tabs (not needed for dev tooling)
- Keyboard navigation for preview (can add later)
- Document text search within preview (can add later)
