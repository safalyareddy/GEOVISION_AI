# Evaluation Protocol & Benchmark Standards

---

## 1. Frozen Evaluation Protocol

To prevent test data leakage and hyperparameter overfitting, evaluation follows a frozen protocol:

1. **Held-Out Partition**: 54 patches reserved in `dataset/test/` never seen during training or validation hyperparameter tuning.
2. **Fixed Thresholding**: Default binary decision threshold set at $\tau = 0.50$.
3. **Metric Definitions**:
   $$\text{Dice} = \frac{2 |P \cap T|}{|P| + |T| + \epsilon}$$
   $$\text{IoU} = \frac{|P \cap T|}{|P \cup T| + \epsilon}$$
   $$\text{Precision} = \frac{|P \cap T|}{|P| + \epsilon}, \quad \text{Recall} = \frac{|P \cap T|}{|T| + \epsilon}$$
   where $P$ is the predicted binary mask, $T$ is the ground-truth mask, and $\epsilon = 10^{-6}$.

---

## 2. Empirical Benchmark Execution

```bash
# Execute evaluation suite across all 3 models and output CSV report
PYTHONPATH=. python src/evaluate.py --save-csv --output reports/test_benchmark_results.csv
```
Detailed results recorded in `reports/heldout_test_evaluation.md`.
