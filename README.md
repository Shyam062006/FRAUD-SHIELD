# FRAUD SHIELD: AI Fraud Detection System

FRAUD SHIELD is an end-to-end machine learning fraud-detection application for college capstone and technical demonstration. Built on a trained Random Forest classifier using the PaySim mobile financial transaction dataset, it connects an interactive high-performance WebGL/GSAP client interface to a FastAPI backend without modifying the UI on disk.

---

## Quick Start (Single Command)

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Start the Server
```bash
# Run from project root:
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# Or run from backend/ directory:
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Open in Browser
Visit **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** to interact with the live model.

---

## Project Structure

```
FRAUD-DETECTION-AI/
├── ML/                       # Machine learning notebooks and serialized artifacts (READ-ONLY)
│   ├── experimentation_and_benchmark.ipynb
│   ├── train_final_model.ipynb
│   ├── model.pkl             # Trained RandomForestClassifier (50 trees)
│   └── scaler.pkl            # Trained StandardScaler (9 features)
├── frontend/                 # Interactive UI interface (READ-ONLY)
│   └── index.html            # Verified byte-for-byte identical via SHA-256
├── backend/                  # FastAPI backend service
│   ├── app/
│   │   ├── main.py           # FastAPI app, static server & in-memory bridge injection
│   │   ├── config.py         # App configuration & settings
│   │   ├── schemas.py        # Pydantic v2 validation models
│   │   ├── features.py       # Domain feature engineering pipeline
│   │   ├── model_service.py  # Model lifecycle and inference manager
│   │   ├── explain.py        # Plain-language explanation engine
│   │   └── metrics.json      # Benchmark results from notebooks
│   ├── static/
│   │   └── bridge.js         # Integration bridge injected in-memory at runtime
│   ├── tests/                # 99 automated pytest test cases
│   ├── requirements.txt      # Pinned dependencies matching training environment
│   └── README.md
└── docs/                     # Comprehensive project documentation
    ├── README.md             # Documentation overview & Mermaid architecture
    ├── API.md                # REST API endpoint reference & schemas
    ├── ARCHITECTURE.md       # Sequence diagrams & data flow
    ├── ML_NOTES.md           # Dataset, feature importances, and limitations
    ├── TESTING.md            # Test suite breakdown & golden test analysis
    ├── DISCOVERY.md          # Phase 0 discovery and audit findings
    └── UI_HASH.txt           # SHA-256 cryptographic verification record
```

---

## Test Suite Execution

Run all 99 automated tests:
```bash
python -m pytest backend/tests/ -v
```

---

## Integrity & Design Principles

1. **Frontend Read-Only Guarantee:** `frontend/index.html` was never modified. Its SHA-256 hash (`56136302EEE27AB79EEB36211C72A8F2D3828BA21B9F0E62CFFC9548E983E214`) is verified and documented in `docs/UI_HASH.txt`.
2. **In-Memory Bridge:** When serving `GET /`, the backend injects `<script src="/bridge.js"></script>` before `</body>` purely in memory.
3. **No Phantom Metrics:** All metrics and feature names correspond strictly to `ML/model.pkl`, `ML/scaler.pkl`, and `ML/experimentation_and_benchmark.ipynb`.
4. **Offline Resilience:** If the backend is stopped, the UI falls back gracefully to local client heuristics without breaking.
