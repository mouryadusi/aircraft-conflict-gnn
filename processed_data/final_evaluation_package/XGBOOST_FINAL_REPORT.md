# Final XGBoost Evaluation

## Model

XGBoost was trained as an edge-level tabular baseline using the six edge features stored in the rebuilt graph dataset.

## Dataset split

- Total graphs: 72,841
- Training graphs: 50,988
- Validation graphs: 10,926
- Test graphs: 10,927

The split was chronological to prevent future information from entering training.

## Validation threshold

- Best threshold: 0.31
- Validation F1: 1.000000

## Test results

| Metric | Result |
|---|---:|
| Precision | 0.998883 |
| Recall | 1.000000 |
| F1 | 0.999441 |
| ROC-AUC | 1.000000 |

## Confusion matrix

```text
[[79480     2]
 [    0  1788]]
```

## Important methodological note

This model is an edge-level tabular baseline. It does not perform message passing or explicitly model aircraft as a graph. Its purpose is to provide a conventional machine-learning reference against which the GCN and GAT models can be assessed.
