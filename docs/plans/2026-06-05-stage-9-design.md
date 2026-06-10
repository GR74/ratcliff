# Stage 9 Design — Decision Trajectory Trace + Phase Diagram

**Date:** 2026-06-05
**Status:** APPROVED (autonomous-mode go-ahead)
**Scope:** Two visualization features. (1) In the 3D Field tab, trace the winning region's trajectory and flash the peak at the moment of commitment. (2) A new Phase Diagram tab: a 2D heatmap sweeping threshold × drift amplitude, colored by accuracy or mean RT.

---

## 1. Motivation

User-requested. Both make the tool more compelling for papers + talks:

- **Trajectory trace**: "this region accumulated evidence and became the decision." Flash the winning peak, draw the winning path, leave a faint trail. Presentation/intuition value.
- **Phase diagram**: x = threshold, y = drift amplitude, color = accuracy or RT. The standard research artifact that reveals parameter regimes and bifurcations at a glance.

Honesty note: this is a 5-category spatial model, not 2-choice. "Accuracy" is defined as the proportion of trials whose decision lands in the **target region (category 1, the innermost zone around UJ1)**. RT is the mean decision time. Both are exposed; the label states the definition.

## 2. Feature A — Decision trajectory trace

### 2.1 Backend (`field_snapshots`, single mode only)

Extend the single-trial field payload with:
- `trajectory`: per-frame `[row, col, value]` of the **argmax location** on the full (N, M) grid, mapped to the downsampled grid coordinates the surface uses. This is "where the leading edge of evidence is" at each frame.
- `crossing_frame`: the index of the first sampled frame at which `max(field) > cr` (the commitment moment), or `null` if it never crosses within the sampled frames.

Mean mode has no single crossing, so trajectory/crossing are omitted there.

### 2.2 Frontend (`FieldView`)

- Draw a `THREE.Line` (or thin tube) through the trajectory points up to the current frame — the "winning path." Older segments fade (lower opacity) → the faint trail.
- A small glowing sphere marker rides the current trajectory point.
- At `crossing_frame`, briefly **flash** the marker (scale + emissive pulse) to signal commitment, and freeze a brighter marker at the winning peak.

## 3. Feature B — Phase diagram

### 3.1 Backend (`phase_diagram` + `/api/phase`)

`phase_diagram(base_params, x_param, x_range, y_param, y_range, grid, nsim, metric)`:
- Sweeps an `grid × grid` mesh over `x_param` (default `cr`, threshold) and `y_param` (default `av1`, drift amplitude).
- At each cell, runs `simulate_b` at `nsim` (default 200) and computes:
  - `accuracy` = proportion of trials in category 1 (target region), OR
  - `rt` = mean RT.
- Since `cr`/`av1` are traced (non-static) JIT args, the simulator compiles **once** and every cell reuses it — no per-cell recompile. The sweep is `grid²` fast calls.
- Returns `{ x_values, y_values, z (grid×grid), x_param, y_param, metric }`.

`/api/phase` endpoint: body `{params, x_param, x_range, y_param, y_range, grid, nsim, metric}`. Defaults: grid=12, nsim=200, x=cr in [4,18], y=av1 in [4,24], metric="accuracy".

CPU cost: ~grid² × per-call. At grid=12 (144 cells), nsim=200, warm: ~1-3 min. On-demand "Generate" button with the honest cost warning.

### 3.2 Frontend (new Phase tab)

- Controls: x-param + range, y-param + range, grid resolution, nsim, metric (accuracy/RT) toggle, "Generate" button.
- Plotly heatmap of `z` with `x_values` / `y_values` axes, a colorbar, and the metric name in the title.
- Reuses Plotly (already bundled) — no new deps.

## 4. Testing

- `field_snapshots` single mode: assert `trajectory` length == n_frames, each entry length 3, `crossing_frame` is int-or-None.
- `phase_diagram`: assert z shape == (grid, grid), values finite, accuracy in [0,1], rt > 0.
- `/api/phase`: happy path + validation errors (bad metric, bad param name).
- Frontend: eyeball via Playwright (3D trail + heatmap render).

## 5. Non-goals

- No bifurcation *detection* (just the heatmap; the eye finds the regimes).
- Trajectory only in single-trial mode.
- Phase diagram limited to the 9 single-condition params for x/y.

## 6. Order

1. Backend: extend `field_snapshots` with trajectory + crossing; tests.
2. Backend: `phase_diagram` + `/api/phase`; tests.
3. Frontend: trail + flash in `FieldView`.
4. Frontend: `PhaseTab` + heatmap + nav wiring.
5. Rebuild, restart, verify both live.
