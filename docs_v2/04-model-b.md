# 04 — Model B: The 2D Gaussian-Random-Field Simulator

## Why Model B is the main model

Model B expands the accumulator to a 100 × 160 spatial field with five response regions. Each timestep adds three signal bumps plus correlated two-dimensional noise. It is computationally demanding: one trial can involve 400 fields of 16,000 cells. The repository’s major performance work is about generating and accumulating those fields efficiently.

## Spatial layout

Three Gaussian drift bumps share row center 50 and use columns 80, 30, and 130. Their amplitudes are \`av1\`, \`av2\`, and \`av3\`. A separate zone map converts the winning cell into a response:

- a tight center around the first bump is category 1;
- its broader ring is category 2;
- regions around the second and third bumps are categories 3 and 4;
- all remaining background is category 5.

~~~mermaid
flowchart LR
    B2["Bump at column 30<br/>category 3"] --- B1["Bump at column 80<br/>category 1 center<br/>category 2 ring"]
    B1 --- B3["Bump at column 130<br/>category 4"]
    BG["Everywhere else<br/>category 5"] -. surrounds .-> B2
    BG -. surrounds .-> B1
    BG -. surrounds .-> B3
~~~

## Exact FFT noise path

\`grf.py\` implements circulant embedding:

1. evaluate the spatial covariance kernel on positive and reflected displacements;
2. embed it in a \`(2N-1) × (2M-1)\` block-circulant matrix;
3. take a 2D FFT to obtain the spectrum;
4. use its square root, \`LAM\`, to color complex Gaussian noise;
5. take another FFT and extract the upper-left 100 × 160 field.

The real and imaginary parts are independent fields, so one complex FFT supplies noise for two timesteps. The embedding is only positive definite for a limited correlation length; the API limits \`sig\` to 17.5 and the fitting clamp uses 17.0.

~~~mermaid
flowchart LR
    Z["Independent normal noise<br/>g1 + i·g2"] --> S["Multiply by spectral<br/>square root LAM"]
    S --> F["Batched 2D FFT"]
    F --> R["Real part<br/>GRF timestep A"]
    F --> I["Imaginary part<br/>GRF timestep B"]
~~~

## Low-rank K–L path

\`grf_kl.py\` offers a Karhunen–Loève approximation. It sorts spectral modes by variance, retains enough modes for the configured threshold (99% by default, capped at 2,000 complex modes), materializes cosine/sine basis columns, and samples fields with one matrix multiplication.

This trades a controlled amount of omitted variance for lower memory traffic and faster GPU fitting. Basis values are cached by rounded \`sig\`. During fitting, the basis is padded to a fixed width so changing \`sig\` does not force a new XLA compilation for every optimizer vertex.

| Path | Generator | Main strength | Main cost |
|---|---|---|---|
| FFT | exact circulant embedding | reference-quality covariance | very large noise/FFT tensors |
| K–L | truncated spectral basis | lower memory and faster GPU evaluation | approximate covariance; basis construction |

## Simulation pipeline

\`simulate.py\` builds all noise for a chunk, interleaves paired fields, adds the three drift bumps, demeans every field, cumulatively sums 400 steps, and finds the first threshold crossing. It returns RT in milliseconds and categories 1–5.

The outer \`simulate_b\` function is not JIT-compiled because KL basis construction uses NumPy. The heavy FFT and KL inner functions are separately JIT-compiled.

## Parameter vector

The fitting objective uses 13 entries:

| Indices | Names | Meaning |
|---:|---|---|
| 0–3 | \`ter, st, cr, crsd\` | nondecision time, its variability, threshold, threshold variability |
| 4–6 | \`sis, sig, sv\` | drift width, noise correlation, legacy variability slot |
| 7–9 | \`av1c1, av2c1, av3c1\` | three drift amplitudes for condition 1 |
| 10–12 | \`av1c2, av2c2, av3c2\` | three drift amplitudes for condition 2 |

\`si\`, the response-zone width, is fixed at 6.0. The \`sv\` slot is clamped and preserved in the 13-value interface but is not consumed by \`simulate_b\`; the frontend recovery table therefore omits index 6 from active parameters.

## High-level wrapper

\`api.py\` isolates JAX details from the web backend. It validates parameters, selects CPU or GPU defaults, caches previews, warms the JIT, converts arrays to lists, fits models, predicts both conditions, produces field snapshots, and computes phase diagrams.

CPU defaults use the FFT path with small chunks. GPU defaults use the KL path with production-scale simulation. That device-aware decision is why the same interface can be interactive locally and much faster on an H100.

## Validation strategy

Tests check covariance embedding validity, GRF variance, KL variance capture, FFT/KL autocorrelation parity, end-to-end output parity, deterministic keys, valid categories, non-multiple chunk sizes, finite objectives, fitting callbacks, API JSON serialization, field trajectories, and phase-diagram shapes.
