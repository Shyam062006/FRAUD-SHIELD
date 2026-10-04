# FRAUD SHIELD Machine Learning Technical Notes

**Document Scope:** Detailed analysis of the dataset, features, model training, evaluation metrics, decision threshold, and system limitations for FRAUD SHIELD.

---

## 1. Dataset Analysis

### 1.1 Dataset Origin: PaySim
The model was trained on the **PaySim** dataset, a multi-agent financial simulator modeled after real mobile money financial transactions implemented by an African mobile financial service company.

- **Raw Dimensions:** 3,479,442 records across 11 features.
- **Transaction Types in Raw Data:** `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`.
- **Target Variable:** `isFraud` (1 = fraudulent transaction, 0 = legitimate).

### 1.2 Domain Filtering
In financial mobile payment systems, fraud almost exclusively occurs when funds are moved between accounts or extracted out of the network:
- `TRANSFER`: Moving money from one customer account to another.
- `CASH_OUT`: Withdrawing cash from an account through a local agent/merchant.

**Analysis from `ML/experimentation_and_benchmark.ipynb` confirmed that 100% of recorded fraud events occur within `TRANSFER` and `CASH_OUT`.**
Filtering out irrelevant transaction types reduced the training domain to **1,527,257 rows**, dramatically improving the signal-to-noise ratio and training efficiency.

---

## 2. Feature Engineering & Importance Analysis

### 2.1 The 9 Production Features

The model expects exactly 9 features in the following strict order:

| # | Feature | Type | Meaning | Production Formula / Source |
|:---:|:---|:---:|:---|:---|
| 0 | `step` | Float | Hour of transaction | Mapped from time `HH:MM` ($0.0 \dots 23.98$) |
| 1 | `amount` | Float | Transaction amount (₹) | User input `txAmt` |
| 2 | `oldbalanceOrg` | Float | Sender starting balance | User input `txSB` |
| 3 | `newbalanceOrig` | Float | Sender ending balance | $\max(0.0, \text{oldbalanceOrg} - \text{amount})$ |
| 4 | `oldbalanceDest` | Float | Recipient starting balance | User input `txRB` |
| 5 | `newbalanceDest` | Float | Recipient ending balance | Post-transaction recipient balance |
| 6 | `errorBalanceOrig` | Float | Sender accounting discrepancy | $\text{newbalanceOrig} + \text{amount} - \text{oldbalanceOrg}$ |
| 7 | `errorBalanceDest` | Float | Recipient accounting discrepancy | $\text{oldbalanceDest} + \text{amount} - \text{newbalanceDest}$ |
| 8 | `type_TRANSFER` | Float | Transaction type indicator | `1.0` if `TRANSFER`, `0.0` if `CASH_OUT` |

### 2.2 Feature Importance Distribution
Inspection of the serialized `RandomForestClassifier` revealed the following Gini feature importances:

```
newbalanceOrig    [██████████████████████████████] 50.91%
errorBalanceOrig  [█████                         ]  9.26%
oldbalanceDest    [█████                         ]  8.51%
amount            [█████                         ]  8.32%
type_TRANSFER     [████                          ]  6.70%
oldbalanceOrg     [███                           ]  5.10%
step              [██                            ]  4.21%
newbalanceDest    [██                            ]  3.52%
errorBalanceDest  [██                            ]  3.46%
```

#### Key Insights:
1. **`newbalanceOrig` accounts for over 50% of the model's split decisions:** In fraudulent activity, malicious actors systematically drain sender accounts completely down to zero. A post-transaction sender balance of zero is the single strongest indicator of compromise.
2. **Engineered error balances provide vital ledger verification:** Discrepancies between theoretical balances and ending balances flag irregular zeroing out of destination accounts (mule sink behavior).
3. **Time of Day (`step`) shifts the decision boundary:** The probability of a cashout transaction is significantly higher during nighttime hours (02:00) than daytime hours (14:00).

---

## 3. Benchmark Metrics & Model Selection

Three candidate architectures were benchmarked in `ML/experimentation_and_benchmark.ipynb` using an 80/20 stratified train-test split:

| Model Architecture | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | ROC-AUC | Production Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.8986 | 0.4283 | 0.5801 | 0.9749 | Rejected (low recall misses >50% of fraud) |
| **Random Forest (Winner) 🏆** | **0.9983** | **0.9896** | **0.9939** | **0.9948** | **Selected (Best Precision, Recall, and F1)** |
| **XGBoost** | 0.9845 | 0.9862 | 0.9853 | 0.9950 | Runner-up (tied on ROC-AUC, lower F1) |

- **Why Random Forest Won:** In financial fraud detection, precision and recall must be balanced. False positives annoy legitimate customers, but false negatives cause direct financial loss. Random Forest achieved a **0.9939 F1-score**, outperforming XGBoost by nearly a full percentage point on fraud recall.

---

## 4. Decision Threshold Analysis

- **Notebook Evaluation Threshold:** Scikit-learn's standard `model.predict()` applies a threshold of **$\tau = 0.50$** ($P(\text{fraud}) \ge 0.50$).
- **UI Hardcoded Visualizer Threshold:** The UI visualizer uses **$\tau = 0.55$** ($P(\text{fraud}) > 0.55$).
- **Implementation Strategy:**
  - The backend defines `DECISION_THRESHOLD = 0.50` by default in `app/config.py`, consistent with the notebook benchmark metrics.
  - The threshold is fully configurable via the `DECISION_THRESHOLD` environment variable, enabling tuning for high-precision or high-recall operational profiles without code changes.

---

## 5. Limitations & Caveats

> [!WARNING]
> **Academic Model Limitations:**
>
> 1. **Synthetic Artifacts:** The PaySim dataset was synthetically generated. Real-world payment fraud involves sophisticated evasion techniques, transaction splitting (smurfing), and behavioral nuances that synthetic models cannot fully replicate.
> 2. **Concept Drift:** Fraud patterns evolve rapidly. As security teams patch vulnerabilities, fraudsters invent new attack vectors. Static machine learning models will inevitably degrade without continuous retraining.
> 3. **Imbalanced Real-World Skew:** Although the dataset reflects an estimated 0.17% fraud rate, real-world banking environments often experience fraud rates of less than 0.05%, amplifying the operational cost of false positives.
> 4. **Disclaimer:** This model is designed for educational demonstrations and college capstone evaluation. It must not be deployed in real-world banking infrastructure without extensive audit, fraud-analyst review loops, and regulatory compliance validation.
