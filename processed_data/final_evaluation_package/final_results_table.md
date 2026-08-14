# Final GCN vs GAT Evaluation

| Model | Features | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| GCN | Trajectory | 0.7792 | 0.9155 | 0.8419 | 0.9984 |
| GAT | Trajectory + CPA | 0.4214 | 1.0000 | 0.5929 | 0.9986 |
| GAT | CPA-only | 0.1354 | 0.9804 | 0.2379 | 0.9680 |
| GAT | No-CPA | 0.4990 | 0.9983 | 0.6654 | 0.9971 |
