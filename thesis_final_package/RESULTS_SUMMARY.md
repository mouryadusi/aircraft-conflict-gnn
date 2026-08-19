# Aircraft Conflict Prediction Using Graph Neural Networks

## Final Model Results

| Model | Features | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| GCN | Trajectory | 0.7792 | 0.9155 | 0.8419 | 0.9984 |
| GAT | Trajectory + CPA | 0.4214 | 1.0000 | 0.5929 | 0.9986 |
| GAT | CPA-only | 0.1354 | 0.9804 | 0.2379 | 0.9680 |
| GAT | No-CPA | 0.4990 | 0.9983 | 0.6654 | 0.9971 |

## Best Performing Model

The GCN trajectory-based model achieved the strongest classification balance with an F1-score of 0.8419.

## Validation Strategy

Models were evaluated using chronological train-validation-test separation to prevent temporal leakage.

## Feature Analysis

CPA variables were analysed as diagnostic features. No-CPA experiments evaluated prediction using observable aircraft relative-state information only.

