# 08 — Cognitive Society

## The research idea

The simulator models how one decision-maker converts evidence into a choice. \`cognitive_society/\` asks what happens when many such decision-makers observe one another. Can a group use confidence and past outcomes to benefit from expertise without being captured by confident but wrong peers?

The package deliberately uses a lightweight NumPy DDM for most experiments. It can run many agents and thousands of decisions in seconds without a JAX compilation or GPU.

## Two conceptual engine tiers

| Engine | Module | Choices | Intended use |
|---|---|---:|---|
| \`DDMAgent\` | \`agent.py\` | 2 | fast society experiments |
| \`SpatialDDMEngine\` | \`engine.py\` | 5 | heavyweight Model B adapter |

\`DecisionEngine\` documents a shared \`decide\`/\`decide_batch\` protocol. The adapter proves Model B can expose that basic interface. However, the integrated \`Society.round\` currently calls DDMAgent-specific \`extra_drift\` and \`boundary_scale\` arguments and assumes binary choices. The full social loop is therefore not yet directly interchangeable with the five-choice spatial adapter without another abstraction layer.

## One lightweight agent

\`DDMParams\` contains boundary, drift scale, nondecision time, and noise. \`DDMAgent.decide_batch\` advances all active trials together at \`DT = 0.01\` seconds until they cross ±boundary or hit 2,000 steps. Evidence sign determines the correct binary choice. A negative drift scale creates a consistently wrong adversary.

## Communication as evidence

\`comms.social_drift\` converts peer decisions into an additive term inside an agent’s DDM:

\[
social\ drift=g\sum_j \bar{w}_j\,leaning_j,\quad
w_j=trust_j\times competence_j
\]

Weights are clipped, normalized, and multiplied by \`social_gain\`, so influence is bounded. A leaning maps choice 0/1 to −1/+1.

\`TrustModel\` is a slower meta-accumulator. Its evidence increases when a peer’s leaning matched the revealed outcome and decreases otherwise. A logistic function maps evidence to trust in (0, 1). Cognitive-map competence can seed the starting trust.

## The round loop

~~~mermaid
flowchart TD
    E["Signed problem evidence"] --> P["PRIVATE<br/>each agent samples 21 decisions"]
    P --> L["Private majority leaning"]
    P --> U["Confidence from vote consistency"]
    L --> S["SOCIAL<br/>peer leanings × trust × competence"]
    U --> G["Uncertainty gate<br/>raise caution and social gain"]
    S --> F["Final DDM decision"]
    G --> F
    F --> M["Collective majority"]
    E --> O["Known outcome"]
    O --> T["Update peer trust"]
    L --> T
~~~

\`SocietyConfig\` creates controlled ablations:

- \`cfg_private\`: no social input;
- \`cfg_flat\`: equal peer influence;
- \`cfg_outcome_trust\`: trust learned from outcomes;
- \`cfg_full\`: competence prior plus uncertainty gating;
- \`cfg_rl\`: learned deference multiplier;
- \`cfg_swarm\`: local graph neighbors and subjective competence maps.

## Cognitive mapping

\`ez_diffusion.py\` recovers approximate 1D drift, boundary, and nondecision time from choices and correct-response RT moments. In \`Society.build_cognitive_maps\`, the operational competence map is simpler: it estimates each agent’s accuracy across configured evidence levels.

Global mode measures one competence vector and copies it to every observer. Local swarm mode lets each observer estimate only its neighbors with a smaller observation budget, producing subjective maps.

## Network topologies

\`topology.py\` returns boolean adjacency matrices for complete, ring-lattice, Watts–Strogatz, random-regular, and Barabási–Albert graphs.

~~~mermaid
flowchart LR
    C["Complete<br/>everyone hears everyone"] --> R["Ring<br/>fixed local neighbors"]
    R --> W["Small world<br/>ring + rewiring"]
    W --> RR["Random regular<br/>same degree, randomized"]
    RR --> BA["Scale free<br/>hub structure"]
~~~

The adjacency row masks which peers influence each agent. Local topology is central to adversarial-cluster experiments because a cluster may be a global minority but a local majority around one victim.

## Reinforcement-learned deference

\`rl.DeferencePolicy\` uses four features: own uncertainty, mean peer trust, peer consensus, and a bias. A linear Gaussian policy samples a deference multiplier, receives 1 for a correct final decision and 0 otherwise, and updates with REINFORCE plus a running baseline. It is intentionally small and interpretable.

## Important modeling assumptions

- Outcomes are revealed after each problem. Outcome-grounded trust cannot work when truth is never observed.
- Private confidence is derived from repeated internal decisions, not a direct neural confidence signal.
- Trust updates receive the same truth and peer leanings for every observer; even in local-map mode, outcome updates are objectively grounded and replicated rather than fully subjective.
- The simulator is a research abstraction of cognition, not evidence that actual people implement the exact algorithm.
- The experiments use known adversary constructions and controlled synthetic environments; real-world transfer requires separate validation.
