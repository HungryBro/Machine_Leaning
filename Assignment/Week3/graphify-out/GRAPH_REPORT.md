# Graph Report - Assignment/Week3  (2026-07-15)

## Corpus Check
- Corpus is ~32,961 words - fits in a single context window. You may not need a graph.

## Summary
- 120 nodes · 197 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Results and Inputs|Results and Inputs]]
- [[_COMMUNITY_Full Lab Analytics|Full Lab Analytics]]
- [[_COMMUNITY_Analytical Functions|Analytical Functions]]
- [[_COMMUNITY_Theory and Slides|Theory and Slides]]
- [[_COMMUNITY_Metrics and Curves|Metrics and Curves]]
- [[_COMMUNITY_Simulation Variables|Simulation Variables]]
- [[_COMMUNITY_Models and Targets|Models and Targets]]
- [[_COMMUNITY_Compact Pipeline|Compact Pipeline]]
- [[_COMMUNITY_Analytical Results|Analytical Results]]
- [[_COMMUNITY_Compact API|Compact API]]
- [[_COMMUNITY_Numerical Integration|Numerical Integration]]
- [[_COMMUNITY_Monte Carlo|Monte Carlo]]
- [[_COMMUNITY_Two-Point Setup|Two-Point Setup]]

## God Nodes (most connected - your core abstractions)
1. `Compact Bias-Variance Lab Script` - 9 edges
2. `simulation` - 8 edges
3. `simulation` - 8 edges
4. `simulation` - 8 edges
5. `Average-Fit Visualization` - 8 edges
6. `Interactive Generalization and Learning-Curve Playground` - 7 edges
7. `Learning Curves` - 7 edges
8. `Constant` - 6 edges
9. `Linear` - 6 edges
10. `Linear through origin` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Average-Fit Visualization` --references--> `Variance Metric`  [INFERRED]
  Assignment/Week3/plots/average_fit.png → Assignment/Week3/README.md
- `analytical_model Function` --semantically_similar_to--> `Browser curveFor Function`  [INFERRED] [semantically similar]
  Assignment/Week3/bias_variance_lab.py → Assignment/Week3/index.html
- `Compact Bias-Variance Lab Script` --references--> `Learning-Curve Noise Levels`  [EXTRACTED]
  Assignment/Week3/bias_variance_lab_compact.py → Assignment/Week3/README.md
- `Compact Bias-Variance Lab Script` --implements--> `Bias-Variance Decomposition`  [EXTRACTED]
  Assignment/Week3/bias_variance_lab_compact.py → Assignment/Week3/README.md
- `Compact Bias-Variance Lab Script` --implements--> `Learning Curves`  [EXTRACTED]
  Assignment/Week3/bias_variance_lab_compact.py → Assignment/Week3/README.md

## Hyperedges (group relationships)
- **Week 3 Bias-Variance Experiment** — sin_pi_x_target, x_squared_target, constant_model, linear_model, linear_through_origin_model, bias_squared_metric, variance_metric, expected_out_error [EXTRACTED 1.00]
- **Learning-Curve Rendering Flow** — learning_curve_playground, html_curve_for_function, html_render_lc_curve_function, learning_curves, expected_out_error [EXTRACTED 1.00]

## Communities (14 total, 3 thin omitted)

### Community 0 - "Results and Inputs"
Cohesion: 0.16
Nodes (24): bias_variance, sin(pi*x), x^2, sigma_0.0, sigma_0.3, learning_curve, sin(pi*x), x^2 (+16 more)

### Community 1 - "Full Lab Analytics"
Cohesion: 0.11
Nodes (19): analytical_constant(), analytical_model(), ConstantModel, fit(), gD_linear(), learning_curve(), LinearModel, LinearOriginModel (+11 more)

### Community 2 - "Analytical Functions"
Cohesion: 0.18
Nodes (11): analytical_constant Function, analytical_model Function, gD Linear Hypothesis Function, gD Linear-Origin Hypothesis Function, Generalization Playground, Browser curveFor Function, Browser fitModel Function, Browser renderGenCmp Function (+3 more)

### Community 3 - "Theory and Slides"
Cohesion: 0.27
Nodes (10): Average Hypothesis g_bar(x), Bias-Variance Decomposition, Normal-Equation Least Squares, simulate_bias_variance Function, Bias-Variance Updated Presentation, Uniform[-1,1] Sampling, Average-Fit Visualization, Full Bias-Variance Lab Script (+2 more)

### Community 4 - "Metrics and Curves"
Cohesion: 0.24
Nodes (10): Bias Squared Metric, Bias-Variance Results Dataset, Expected Out-of-Sample Error Eout, Browser renderLcCurve Function, Learning-Curve Noise Levels, Learning-Curve Results Dataset, Learning Curves, Variance Metric (+2 more)

### Community 5 - "Simulation Variables"
Cohesion: 0.50
Nodes (9): simulation, simulation, simulation, bias2, eout, g_bar, std, variance (+1 more)

### Community 6 - "Models and Targets"
Cohesion: 0.48
Nodes (7): Constant Model, Linear Model, Linear Through Origin Model, Target Function sin(pi*x), Compact Bias-Variance Lab Script, Interactive Generalization and Learning-Curve Playground, Target Function x^2

### Community 8 - "Compact Pipeline"
Cohesion: 0.40
Nodes (6): Compact fit_predict Function, Compact learning_curve Function, Compact Plotting Pipeline, Compact simulate Function, Full learning_curve Function, Full Lab Results Pipeline

### Community 9 - "Analytical Results"
Cohesion: 0.60
Nodes (6): bias2, eout, variance, analytical, analytical, analytical

### Community 10 - "Compact API"
Cohesion: 0.83
Nodes (3): fit_predict(), learning_curve(), simulate()

## Knowledge Gaps
- **12 isolated node(s):** `ConstantModel`, `LinearModel`, `LinearOriginModel`, `Uniform[-1,1] Sampling`, `Analytical Numerical Integration` (+7 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Browser curveFor Function` connect `Analytical Functions` to `Theory and Slides`, `Metrics and Curves`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Why does `analytical_model Function` connect `Analytical Functions` to `Compact Pipeline`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `Compact Bias-Variance Lab Script` connect `Models and Targets` to `Compact Pipeline`, `Theory and Slides`, `Metrics and Curves`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **What connects `ConstantModel`, `LinearModel`, `LinearOriginModel` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Full Lab Analytics` be split into smaller, more focused modules?**
  _Cohesion score 0.11067193675889328 - nodes in this community are weakly interconnected._