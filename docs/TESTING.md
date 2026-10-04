# FRAUD SHIELD Test Suite Documentation

**Framework:** Pytest 9.1+ with FastAPI TestClient  
**Total Automated Tests:** 70  
**Test Suite Status:** 70 Passed, 0 Failed, 0 Skipped  

---

## 1. Test Architecture & Structure

The test suite is located under [backend/tests/](file:///c:/Users/siddh/OneDrive/Desktop/FRAUD-DETECTION-AI/backend/tests/) and divided into dedicated test modules:

```
backend/tests/
├── conftest.py            # Global test fixtures and sys.path configuration
├── test_features.py       # Unit tests for feature mapping and time parsing
├── test_validation.py     # Pydantic boundary, range, and format enforcement
├── test_explain.py        # Plain-language explanation generation and severity sorting
├── test_threshold.py      # Classification threshold and verdict determination
├── test_golden.py         # Golden equivalence comparing API risk to raw model calls
└── test_api.py            # Comprehensive endpoint, boundary, and sample scenario tests
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

---

## 4. Honest Sample Comparison Report

We evaluated the two sample buttons configured in the UI:

| Sample Scenario | Transaction Parameters | API Output Probability | Verdict ($\tau=0.50$) | Contributing Signals |
|:---|:---|:---:|:---:|:---|
| **Sample Legit** | `TRANSFER`, ₹250, Sender ₹5,000, Recipient ₹3,200, 14:30 | **`0.0000` (0.0%)** | `CLEARED` | All normal (Level 0). Small amount relative to balance, normal business hour, standard transfer. |
| **Sample Fraud** | `CASH_OUT`, ₹9,800, Sender ₹10,000, Recipient ₹0, 02:15 | **`0.3200` (32.0%)** | `CLEARED` (Elevated Warning) | Level 2 Warnings: Moving 97% of balance in one go, recipient account empty, transacted late at night. Level 1 Warning: Cash-out withdrawal. |

### Technical Observations:
- **Sample Legit** evaluates to absolute zero risk (`0.0000`).
- **Sample Fraud** jumps to `0.3200` (32% risk), representing a massive relative elevation.
- Under the benchmark decision threshold ($\tau = 0.50$), `0.3200` produces `CLEARED` because the sender's account was not 100% drained (₹200 remained). When the amount is increased to ₹10,000 (100% account drainage), the model output jumps to **`0.7200` (72% risk)** and triggers **`FLAGGED`**.
- This transparently mirrors the true behavior of the Random Forest model and its 50.91% feature importance on `newbalanceOrig`.
