# FINAL RESEARCH FREEZE

Project: AI-Enabled Aircraft Conflict Prediction

Status: RESEARCH / MODEL FREEZE

Purpose:
Freeze the exact computational state used for the dissertation results.
No new model training should occur after this point unless explicitly
approved as a separate experiment.

## Final validated model results

| Model | Feature configuration | F1 | ROC-AUC |
|---|---|---:|---:|
| Rule-based | ICAO threshold | 0.0955 | — |
| XGBoost | Observable-state engineered features | 0.7069 | 0.9998 |
| GCN | Trajectory-only | 0.8419 | 0.9984 |
| GAT | Observable state / No-CPA | 0.6654 | 0.9971 |
| GAT | Observable state + CPA | 0.5929 | 0.9986 |
| GAT | CPA-only | 0.2379 | 0.9680 |

## Dataset split audit

Train:
- Graphs: 50,988
- Nodes: 293,010
- Edges: 430,168
- Positive edges: 8,552
- Positive rate: 0.0198806
- Time: 1496620801–1496681957

Validation:
- Graphs: 10,926
- Nodes: 95,652
- Edges: 162,748
- Positive edges: 3,218
- Positive rate: 0.0197729
- Time: 1496681958–1496693254

Test:
- Graphs: 10,927
- Nodes: 54,833
- Edges: 81,270
- Positive edges: 1,788
- Positive rate: 0.0220007
- Time: 1496693255–1496707199

## Important methodological decision

CPA-derived features are retained as diagnostic ablations.

The primary production formulation uses observable trajectory/state information
rather than CPA-derived projected future geometry.

## Outstanding verification

DCPA diagnostic AUC must be confirmed as TEST-SET-ONLY before making a
specific scientific claim about its held-out predictive performance.

## Freeze rule

Do not:
- retrain the primary models
- alter the final labels
- alter the train/validation/test split
- change feature definitions
- overwrite final metrics
- silently replace figures
- silently replace experiment results

Any new experiment must receive a new experiment ID and be explicitly
labelled as POST-FREEZE.

## Environment

See:
- environment.yml
- pip_freeze.txt
- software_versions.txt

## Source integrity

See:
- SHA256SUMS.txt

