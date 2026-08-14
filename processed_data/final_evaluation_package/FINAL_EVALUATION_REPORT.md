# Final GCN / GAT Evaluation Report

## 1. Evaluation scope

The final large-dataset evaluation compares four graph neural network configurations under the same chronological train/validation/test framework:

1. **GCN — Trajectory**
2. **GAT — Trajectory + CPA**
3. **GAT — CPA-only**
4. **GAT — No-CPA**

The evaluation uses a chronological split rather than a random split, preserving temporal separation between training, validation and test data.

## 2. Corrected final results

| Model | Features | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| GCN | Trajectory | 0.7792 | 0.9155 | 0.8419 | 0.9984 |
| GAT | Trajectory + CPA | 0.4214 | 1.0000 | 0.5929 | 0.9986 |
| GAT | CPA-only | 0.1354 | 0.9804 | 0.2379 | 0.9680 |
| GAT | No-CPA | 0.4990 | 0.9983 | 0.6654 | 0.9971 |

## 3. Best-performing configuration by F1

The highest test F1 is obtained by **GCN — Trajectory**, with F1 = **0.8419**.

Its precision is **0.7792**, recall is **0.9155**, and ROC-AUC is **0.9984**.

## 4. Best ROC-AUC

The highest ROC-AUC is obtained by **GAT — Trajectory + CPA**, with ROC-AUC = **0.9986**.

## 5. DCPA / CPA diagnostic

The single-feature DCPA diagnostic produced **ROC-AUC = 0.877048**.

This demonstrates that DCPA is strongly predictive of the conflict label. However, the result alone does not establish deterministic label leakage.

The CPA-only GAT achieved:

- Precision: **0.1354**
- Recall: **0.9804**
- F1: **0.2379**
- ROC-AUC: **0.9680**

The CPA-only experiment therefore confirms that CPA geometry contains substantial predictive signal, but CPA-only information does not reproduce the performance of the strongest trajectory-based model.

## 6. No-CPA configuration

The No-CPA GAT removes `tcpa` and `dcpa` from the edge representation.

The remaining four edge features represent currently observable relative-state information:

- Horizontal separation
- Vertical separation
- Relative speed
- Relative heading

The No-CPA GAT achieved:

- Precision: **0.4990**
- Recall: **0.9983**
- F1: **0.6654**
- ROC-AUC: **0.9971**

This configuration is the principal candidate for the observable-state-only formulation.

## 7. Dataset statistics

- Graphs: **72,841**
- Directed edges: **674,186**
- Positive labels: **13,558**
- Positive rate: **0.020110**
- Node feature dimension: **6**
- Original edge feature dimension: **6**
- CPA candidate rows: **337,093**
- CPA positive labels: **6,779**

## 8. CPA-to-graph label consistency

The CPA-to-graph consistency check confirmed:

- CPA rows: **337,093**
- CPA positive labels: **6,779**
- Graph directed edges: **674,186**
- Graph positive labels: **13,558**
- Expected graph positives after storing each CPA pair in both directions: **13,558**
- Positive-label agreement: **True**

This confirms that the graph construction preserved the CPA conflict labels consistently when converting undirected candidate pairs into directed graph edges.

## 9. Interpretation

The conflict class represents approximately 2% of graph edges. Consequently, ROC-AUC should not be used as the sole indicator of classification quality.

F1, precision and recall provide the more informative view of the operational classification trade-off, while ROC-AUC is retained as a ranking/discrimination measure.

The corrected GCN trajectory model achieves the highest F1 among the evaluated configurations, while the Trajectory + CPA GAT obtains the highest ROC-AUC.

The No-CPA GAT remains highly discriminative and provides the most appropriate formulation for assessing prediction from observable relative-state information without explicit CPA variables.

## 10. Methodological note

The CPA-inclusive experiments should be presented as diagnostic ablations rather than being interpreted automatically as evidence of superior real-time conflict prediction.

CPA variables encode predicted closest-approach geometry and therefore differ methodologically from purely observable current-state features.

## 11. Reproducibility

This packaging script does not retrain any model. It operates on the completed saved evaluation results and the large graph dataset.
