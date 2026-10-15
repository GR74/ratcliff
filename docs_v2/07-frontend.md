# 07 — Frontend: React, State, and Visuals

## Architecture

The frontend is a Vite React TypeScript application. \`main.tsx\` mounts \`App\`; \`App.tsx\` lazy-loads six tabs so Plotly and three.js are not part of the initial bundle. Tailwind supplies layout utilities, Zustand holds shared parameters, and \`lib/api.ts\` is the typed HTTP boundary.

~~~mermaid
flowchart TB
    APP["App.tsx<br/>tab shell"] --> SIM["ForwardSimTab"]
    APP --> FIELD["FieldTab"]
    APP --> PHASE["PhaseTab"]
    APP --> FIT["FitTab"]
    APP --> PRED["PredictTab"]
    APP --> COMP["CompareTab"]
    STORE["Zustand store<br/>params + current tab + GPU URL"] --> SIM
    STORE --> FIELD
    STORE --> PHASE
    STORE --> FIT
    API["lib/api.ts<br/>fetch + error handling"] --> SIM
    API --> FIELD
    API --> PHASE
    API --> FIT
    API --> PRED
    API --> COMP
~~~

## Shared state and types

\`lib/types.ts\` defines the nine-parameter single-condition \`ParamSet\`, 13-entry \`FullParams\`, default values, slider metadata, and every API response shape. \`store.ts\` keeps:

- the current tab;
- the shared nine simulation parameters;
- actions to change/reset parameters;
- the optional GPU endpoint string.

The store is in-memory. Named parameter configurations are separately persisted in \`localStorage\` under \`ratcliff_configs_v1\`.

## The six tabs

### Forward Sim

\`ForwardSimTab.tsx\` watches shared parameters. After a 200 ms debounce, it aborts the previous request and requests a small preview. “Run full” requests 9,000 trials. \`RTHistogram\` overlays Plotly histograms for categories 1–5.

### Field

\`FieldTab.tsx\` asks for 48 snapshots from one trial or the mean of 64 trials. \`FieldView.tsx\` builds a three.js plane whose vertex heights and colors update per frame. It adds orbit controls, an optional threshold plane, and—in single mode—a growing trajectory and commitment marker.

The scene is rebuilt only when the field payload changes. Frame updates reuse the geometry, which is much cheaper than reconstructing the renderer 12 times per second.

### Phase

\`PhaseTab.tsx\` selects two parameters, ranges, grid size, trial count, and either target-category accuracy or mean RT. It displays the returned matrix as a Plotly heatmap. The backend computes grid cells sequentially, so a 24 × 24 grid can be costly.

### Fit

\`FitTab.tsx\` uploads data, chooses a maximum iteration count, optionally accepts a GPU URL, starts a job, and polls every two seconds. \`FitProgress\` draws loss on a logarithmic axis. \`RecoveryTable\` compares returned parameters with \`DEFAULT_FULL_PARAMS\`; this is meaningful for synthetic recovery demonstrations, not as “true values” for arbitrary human uploads.

### Predict

\`PredictTab.tsx\` exposes all 13 fit parameters, runs one or two conditions, and displays response proportions and RT histograms.

### Compare

\`CompareTab.tsx\` reads two named configurations from browser storage, simulates both at 512 trials, and renders separate histograms. Despite the word “overlay” in older documentation, the current UI displays two stacked plots rather than one combined overlay.

## Browser-side data flow

~~~mermaid
sequenceDiagram
    participant User
    participant Component
    participant Store as Zustand/localStorage
    participant API as lib/api.ts
    participant Backend
    User->>Component: move slider
    Component->>Store: setParam
    Store-->>Component: new params
    Component->>Component: debounce 200 ms
    Component->>API: postSimulate(params, signal)
    API->>Backend: fetch /api/simulate
    Backend-->>API: JSON
    API-->>Component: typed result
    Component-->>User: update Plotly chart
~~~

## Development commands

~~~powershell
npm ci --prefix frontend
npm run dev --prefix frontend
npm run build --prefix frontend
npm run preview --prefix frontend
~~~

The production build runs \`tsc -b\` before Vite, so it is the current type-check gate.

## Current limitations

- No frontend unit or browser-test framework is configured.
- React Query is installed but the current API calls use direct \`fetch\`.
- The twelve-column layouts use fixed column spans without responsive breakpoints, so small-screen behavior is limited.
- Importing saved config JSON checks only that the root is an array; it does not validate each object.
- Accessibility is partial: controls generally use native elements, but several labels are not connected with \`htmlFor\`, and canvas/WebGL content has no text equivalent.
- Error states are local to tabs and there is no application-wide error boundary.
