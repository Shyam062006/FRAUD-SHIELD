# FRAUD SHIELD Test Suite Documentation

**Framework:** Pytest 9.1+ with FastAPI TestClient  
**Total Automated Tests:** 99  
**Test Suite Status:** 99 Passed, 0 Failed, 0 Skipped  

---

## 1. Test Architecture & Structure

The test suite is located under [`backend/tests/`](../backend/tests/) and divided into dedicated test modules:

```
backend/tests/
├── conftest.py                      # Global test fixtures and sys.path configuration
├── test_features.py                 # Unit tests for feature mapping and time parsing
├── test_validation.py               # Pydantic boundary, range, and format enforcement
├── test_explain.py                  # Plain-language explanation generation and severity sorting
├── test_threshold.py                # Classification threshold and verdict determination
├── test_golden.py                   # Golden equivalence comparing API risk to raw model calls
├── test_api.py                      # Comprehensive endpoint, boundary, and sample scenario tests
├── test_dataset_benchmark_rows.py   # 10 real fraud + 10 real legit PaySim dataset rows side-by-side
└── test_reason_consistency.py       # 26 consistency checks across >=20 diverse inputs
```

---

## 2. Running the Test Suite

```bash
# From workspace root:
python -m pytest backend/tests/ -v

# Or from backend directory:
cd backend
python -m pytest tests/ -v
```

---

## 3. Test Modules Breakdown

### 3.1 Feature Extraction Tests (`test_features.py`)
- **Feature Count & Ordering:** Validates that the pipeline outputs strictly 9 features in the exact training sequence:
  `['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig', 'errorBalanceDest', 'type_TRANSFER']`.
- **Time Conversion:** Asserts correct conversion of `"HH:MM"` format into fractional hours (`"14:30"` $\rightarrow 14.5$, `"02:15"` $\rightarrow 2.25$).
- **Accounting Invariants:** Verifies balance discrepancy calculations (`errorBalanceOrig` and `errorBalanceDest`) for standard transfers, cash-outs, and overdraft edge cases.

### 3.2 Input Validation Tests (`test_validation.py`)
- **Amount Boundary Values:**
  - Valid: ₹1.0, ₹50.0, ₹100,000.0, ₹1,000,000.0.
  - Rejections (422): ₹0.99, ₹0.0, ₹-10.0, ₹1,000,000.01, ₹2,000,000.0.
- **Account Balance Boundaries:**
  - Valid: ₹0.0 to ₹10,000,000.0.
  - Rejections (422): Negative balances and balances exceeding ₹10,000,000.
- **Transaction Types:**
  - Permitted: `"TRANSFER"`, `"CASH_OUT"`.
  - Rejections (422): `"PAYMENT"`, `"DEBIT"`, `"CASH_IN"`, lowercase `"transfer"`.
- **Time of Day Format:**
  - Valid: `"00:00"`, `"02:15"`, `"14:30"`, `"23:59"`.
  - Rejections (422): `"24:00"`, `"12:60"`, `"9:00"`, `"abc"`, empty string.

### 3.3 Explainability Tests (`test_explain.py`)
- **Signal Completeness:** Ensures every prediction generates reasons covering all 4 defined signals: `balance_behavior`, `time_pattern`, `transaction_type`, and `amount_pattern`.
- **Severity Ranking:** Verifies that reasons are sorted in descending order of severity (`level 2` $\rightarrow$ `level 1` $\rightarrow$ `level 0`).
- **Data Privacy & User-Friendliness:** Asserts that internal column identifiers (such as `newbalanceOrig` or `errorBalanceDest`) are strictly absent from all user-facing strings.
- **Behavioral Triggers:** Asserts that moving $\ge 80\%$ of balance or transacting at 02:15 triggers a Level 2 warning.

### 3.4 Decision Threshold Tests (`test_threshold.py`)
- Asserts default decision threshold is `0.50`.
- Validates verdict transitions:
  - $P < \tau \implies \text{CLEARED}$
  - $P \ge \tau \implies \text{FLAGGED}$
  - Boundary exact equality ($P = 0.50 \implies \text{FLAGGED}$)

### 3.5 Golden Equivalence Tests (`test_golden.py`)
Directly loads `ML/model.pkl` and `ML/scaler.pkl` using joblib. For 5 fixed, distinct transaction vectors, executes:
1. Feature extraction $\rightarrow$ `scaler.transform()` $\rightarrow$ `model.predict_proba()`
2. HTTP POST to `/api/predict`
3. Asserts that the API probability matches the raw model output within a floating-point tolerance of $1 \times 10^{-4}$.

### 3.6 API & Integration Tests (`test_api.py`)
- `GET /api/health`: Confirms `model_loaded: true`, model name, version, and valid uptime.
- `GET /api/model-info`: Confirms feature count (9), feature names, and benchmark metrics matching notebook outputs.
- `GET /`: Confirms HTML delivery with in-memory injected `<script src="/bridge.js"></script>`.
- `GET /bridge.js`: Confirms JavaScript delivery.
- Boundary and 422 input validation assertions across diverse malformed inputs.
- Validates that cleared sub-threshold transactions never produce Level 2 warnings.

### 3.7 Dataset Benchmark Parity Tests (`test_dataset_benchmark_rows.py`)
- Evaluates 10 ground-truth FRAUD rows and 10 ground-truth LEGIT rows extracted directly from the PaySim training dataset.
- Executes each row through both:
  1. Direct notebook pipeline: `model.predict_proba(scaler.transform(df))`
  2. Live API pipeline: `extract_features(PredictRequest) -> scaler.transform -> model.predict_proba`
- Asserts that both probabilities match exactly with $\Delta < 1 \times 10^{-4}$ across all 20 rows ($100\%$ parity).

### 3.8 Explanation Consistency & Grounding Tests (`test_reason_consistency.py`)
- Tests 25 diverse scenarios spanning edge cases, micro-transactions, overdrafts, and peak/off-peak hours.
- **Invariant 1:** If verdict is `CLEARED`, NO reason may be Level 2 (Red).
- **Invariant 2:** If `amount > sender_balance`, verdict is strictly `FLAGGED` with Level 2 reason `"Amount is larger than the sender's balance."`
- **Invariant 3:** A reason may be Level 2 or 1 ONLY if that signal pushes risk UP ($\phi > 0.01$ via Saabas tree path decomposition).
- **Invariant 4:** Explanations always cover all 4 telemetry signals, sorted descending by severity.
- **Invariant 5:** No generic filler phrases (*"within standard clearing parameters"*) are permitted.

---

## 4. Honest Sample Comparison Report

We evaluated key operational scenarios:

| Sample Scenario | Transaction Parameters | API Output Probability | Verdict ($\tau=0.50$) | Contributing Signals |
|:---|:---|:---:|:---:|:---|
| **Sample Legit** | `TRANSFER`, ₹250, Sender ₹5,000, Recipient ₹3,200, 14:30 | **`0.0000` (0.0%)** | `CLEARED` | All normal (Level 0 - Gray). Small amount relative to balance, normal business hour, standard transfer. |
| **Sample Fraud** | `CASH_OUT`, ₹9,800, Sender ₹10,000, Recipient ₹0, 02:15 | **`0.3200` (32.0%)** | `CLEARED` (Elevated Caution) | Level 1 (Amber): Moving 97% of balance in one go, transacted during off-peak morning hours. Level 0 (Gray): Amount and cash-out profile. Strictly no Level 2 since verdict is CLEARED. |
| **Bug Input / Overdraft** | `TRANSFER`, ₹500,000, Sender ₹50,000, Recipient ₹0, 14:30 | **`0.8500` (85.0%)** | `FLAGGED` | Level 2 (Red): *"Amount is larger than the sender's balance."* and staging vector. Level 0 (Gray): Normal business hour, typical amount. |

### Technical Observations:
- **Sample Legit** evaluates to absolute zero risk (`0.0000`).
- **Sample Fraud** jumps to `0.3200` (32% risk), representing an elevated risk signal while remaining sub-threshold (`CLEARED`), producing Amber warnings without violating Level 2 invariants.
- **Overdraft / Anomaly**: Attempting to move ₹500,000 with a balance of ₹50,000 is flagged as an anomaly, producing 85% risk with a Level 2 warning on balance behavior.

