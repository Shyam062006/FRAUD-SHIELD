# FRAUD SHIELD Backend

Production-grade FastAPI backend for the FRAUD SHIELD financial fraud detection system. Serves predictions from a pre-trained Random Forest model trained on the PaySim dataset.

## Architecture

- **Framework:** FastAPI with Uvicorn
- **ML Runtime:** Scikit-Learn (`RandomForestClassifier`, 50 estimators) & `StandardScaler`
- **Data Validation:** Pydantic v2 schemas with boundary enforcement
- **Explanation Engine:** Plain-language telemetry signals mapped from model feature weights and transaction attributes
- **Static Serving:** In-memory bridge injection for `frontend/index.html` (the original HTML on disk is completely read-only)

## Quick Start

### 1. Requirements
Ensure Python 3.11+ is installed.

```bash
cd backend
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` if custom paths or thresholds are needed:
```bash
cp .env.example .env
```

### 3. Run Development Server
```bash
# From backend/ directory:
uvicorn app.main:app --reload --port 8000

# Or from project root:
uvicorn backend.app.main:app --reload --port 8000
```

Open [http://localhost:8000/](http://localhost:8000/) in your browser.

## API Endpoints

- `GET /`: Serves the FRAUD SHIELD UI with runtime-injected bridge script.
- `POST /api/predict`: Evaluates transaction risk.
  - Body: `{ "type": "TRANSFER"|"CASH_OUT", "amount": 1..1000000, "sender_balance": 0..10000000, "recipient_balance": 0..10000000, "time": "HH:MM" }`
  - Returns: `{ "risk": float, "verdict": "FLAGGED"|"CLEARED", "threshold": float, "reasons": [...], "model": str, "latency_ms": float }`
- `GET /api/health`: Health status, model readiness, uptime.
- `GET /api/model-info`: Model specs and PaySim benchmark metrics.
