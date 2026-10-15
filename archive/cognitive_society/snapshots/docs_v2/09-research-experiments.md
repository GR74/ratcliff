# 09 — Research Experiments and Claims

## How to read this research

The \`cognitive_society/\` modules mix simulations, real-data analyses, baselines, and experiment runners. A result from a controlled NumPy simulation is evidence about that constructed system, not automatically about humans or deployed AI. The repository’s strongest practice is that many modules state their honest novelty boundary and retain null or qualified findings.

## Research map

~~~mermaid
flowchart TB
    CORE["Core society<br/>agent + comms + society"] --> ANCHOR["Anchor adversary experiment"]
    CORE --> SWARM["Local-network swarm"]
    CORE --> RL["Learned deference"]
    WOC["Confidence-weighted cognition"] --> B1["Brick 1<br/>one-shot pooling"]
    WOC --> B2["Brick 2<br/>iterated belief loop"]
    WOC --> B3["Brick 3<br/>n-dimensional experts"]
    B3 --> AI["AI applications<br/>routing + labels"]
    AI --> REAL["Real-data checks<br/>RouterBench + crowd data"]
~~~

## Anchor society experiment

\`experiment.py\` creates varied honest agents plus low-boundary, negative-drift adversaries that answer quickly and incorrectly. It compares private, flat, outcome-trust, and full configurations.

The repository README reports approximately 68.2% private, 74.2% flat, 93.8% outcome-trust, and 97.4% full accuracy for its specified 4-honest/3-adversary experiment. The main interpretation is not that “97.4% is universal.” It is that outcome history supplies most of the gain, while cognitive-map initialization and uncertainty gating add a smaller improvement in that environment.

Run:

~~~powershell
python -m cognitive_society.demo
python -m cognitive_society.experiment
python -m cognitive_society.rl_experiment
~~~

## Decentralized swarm sequence

| Module | Question |
|---|---|
| \`swarm_experiment.py\` | Does local topology alone damage an all-honest society? |
| \`adversary.py\` | How do we place a connected, confident, wrong cluster and verify local majority capture? |
| \`baselines.py\` | How do W-MSR, A-RepC, cross-inhibition, no-trust, and oracle comparators behave? |
| \`swarm_metrics.py\` | How accurately and quickly does each observer identify adversarial neighbors? |
| \`phase2_experiment.py\` | Global cluster stress test and ablations |
| \`phase2_per_victim.py\` | Score the honest nodes actually captured by a local adversary majority |
| \`phase3_scale.py\` | Does the local isolation result persist as network size grows? |
| \`phase3_tipping.py\` | Does trust move the collective tipping fraction? |
| \`phase4_consensus.py\` | What changes when opinions propagate iteratively rather than once? |

The narrow demonstrated Phase 2 result is per captured victim: outcome-grounded trust can rank the local colluding cluster correctly while agreement-grounded trust can invert because attackers form the victim’s local majority. Global accuracy is often near-perfect for every method because the adversary cluster remains a global minority; that is why per-victim AUC is the appropriate metric.

~~~mermaid
flowchart LR
    G["Global minority<br/>adversary cluster"] --> V["Boundary victim"]
    V --> L{"Local neighborhood"}
    L --> A["Attackers are local majority"]
    A --> AG["Agreement trust:<br/>wrong consensus looks reliable"]
    A --> OT["Outcome trust:<br/>wrong history loses trust"]
~~~

## Collective-cognition “bricks”

### \`confidence_woc.py\`

Agents estimate a continuous truth and report empirical precision as confidence. It tests when confidence weighting approximates inverse-variance optimal pooling and how overconfident bias breaks it.

### \`confidence_woc_ddm.py\`

Uses the actual binary DDMAgent. Its important correction is that naive inverse reaction time is not reliable confidence when caution and fast errors are confounds. Repeated-decision consistency is the better evidence-state signal in this setup.

### \`belief_loop.py\`

Moves from one-shot pooling to repeated precision-weighted belief exchange on a small-world graph. It studies double-counted evidence, overconfidence, delusion propagation, outcome trust, robust precision, and homeostatic diversity.

### \`confidence_woc_nd.py\`

Generalizes beliefs to vectors. Different agents specialize in different dimensions, and per-dimension confidence acts as a self-gating mixture-of-experts router.

## AI-facing companions

- \`model_routing.py\` simulates choosing among specialist models by confidence or learned task-type track record.
- \`label_aggregation.py\` simulates majority, confidence-weighted, and gold-set track-record aggregation for crowdsourced/RLHF labels.
- \`router_bench.py\` evaluates track-record and cost-aware routing on real RouterBench LLM outputs. Current documentation reports that routing ties a dominant GPT-4 generalist on raw accuracy, improves over comparable open peers, and can improve the quality/cost Pareto frontier. The dataset lacks model confidence, so the calibration-trap portion is not a real-data test.

## Real and semi-real analyses

- \`real_confidence_replication.py\` downloads human confidence datasets from OSF and tests calibration and shared-item pooling. It needs internet and pandas.
- \`real_label_aggregation.py\` uses crowd-kit’s real \`relevance-2\` workers, then injects a synthetic perfectly colluding bloc for the adversarial condition. Clean-data results are real; the attack is semi-real.

These dependencies are not declared in the main \`pyproject.toml\`, so install them in a separate reproducibility environment rather than assuming the base project is sufficient.

## Simulation-based inference track

\`scripts/sbi_mvp.py\` generates Model B simulations, trains an SNPE neural posterior, and checks recovery/calibration. The recorded N=3,000 run reports good recovery for several location/drift parameters and weak recovery for variability parameters.

One especially important code-level caveat: the current Model B simulator does not consume the \`sv\` parameter even though the SBI prior includes a 13th slot. Consequently \`sv\` is structurally unidentifiable in the current implementation, not merely difficult to estimate. Any future SBI study should either activate a scientifically defined \`sv\` effect or remove that dimension.

## Reproduction commands

~~~powershell
python -m cognitive_society.confidence_woc
python -m cognitive_society.confidence_woc_ddm
python -m cognitive_society.belief_loop
python -m cognitive_society.confidence_woc_nd
python -m cognitive_society.model_routing
python -m cognitive_society.label_aggregation
python -m cognitive_society.swarm_experiment
python -m cognitive_society.phase2_per_victim
~~~

Record seed, dependency versions, data download date, and exact command for every reproduced result.
