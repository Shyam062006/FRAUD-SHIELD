# FRAUD SHIELD System Architecture & Data Flow

This document details the architecture, component interaction, and data flows of the FRAUD SHIELD application.

---

## 1. High-Level Architecture Overview

FRAUD SHIELD consists of three decoupled layers:
1. **Frontend Presentation Layer (`frontend/index.html`):** A pre-rendered, high-fidelity WebGL/GSAP client interface. This layer is strictly read-only on disk.
2. **Integration Bridge (`backend/static/bridge.js`):** Injected in-memory by FastAPI at response time. Intercepts UI evaluation calls, queries the API, and updates DOM elements using identical CSS classes.
3. **ML Backend Service (`backend/app/`):** A FastAPI microservice that extracts domain features, scales vectors, evaluates predictions via a Random Forest classifier, and synthesizes plain-language explanations.

```mermaid
graph TB
    subgraph Browser ["Client Browser"]
        DOM["HTML DOM (#trymodel, #drivers, #gaugeArc)"]
        UI_JS["Original UI Logic (renderSim, parseCurrency)"]
        Bridge["Integration Bridge (bridge.js)"]
        DOM <--> UI_JS
        UI_JS <--> Bridge
    end

    subgraph Backend ["FastAPI Backend (Port 8000)"]
        FastAPIApp["main.py (App & Static Server)"]
        InMemInjector["In-Memory Injection Engine"]
        Validator["schemas.py (Pydantic V2)"]
        FeatExtract["features.py (Feature Pipeline)"]
        ModelSvc["model_service.py (Inference Engine)"]
        ExplainEngine["explain.py (Explanation Engine)"]
        MetricsStore["metrics.json (Benchmark Store)"]

        FastAPIApp --> InMemInjector
        FastAPIApp --> Validator
        Validator --> FeatExtract
        FeatExtract --> ModelSvc
        ModelSvc --> ExplainEngine
        FastAPIApp --> MetricsStore
    end

    subgraph Storage ["Read-Only Disk Artifacts"]
        RawHTML["frontend/index.html (Read-Only)"]
        ModelPkl["ML/model.pkl (Random Forest)"]
        ScalerPkl["ML/scaler.pkl (StandardScaler)"]

        InMemInjector -. Read File .-> RawHTML
        ModelSvc -. Unpickle .-> ModelPkl
        ModelSvc -. Unpickle .-> ScalerPkl
    end

    Bridge -- "HTTP POST /api/predict" --> FastAPIApp
```

---

## 2. In-Memory Bridge Injection Flow

To comply with the strict rule that `frontend/index.html` cannot be altered on disk, the backend serves the UI dynamically:

```mermaid
sequenceDiagram
    autonumber
    actor User as User Browser
    participant API as FastAPI (main.py)
    participant Disk as frontend/index.html (Disk)

    User->>API: GET /
    API->>Disk: Read file contents (read-only)
    Disk-->>API: Raw HTML string
    Note over API: In-Memory string replacement:<br/>Replace '</body>' with<br/>'<script src="/bridge.js"></script></body>'
    API-->>User: 200 OK (HTML with bridge script)
    User->>API: GET /bridge.js
    API-->>User: 200 OK (bridge.js content)
    Note over User: UI attaches bridge hooks seamlessly
```

---

## 3. End-to-End Prediction Sequence

When a user clicks **"EVALUATE TRANSACTION"** or either of the sample buttons (**"SAMPLE LEGIT"** / **"SAMPLE FRAUD"**):

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant DOM as UI Elements (DOM)
    participant Bridge as bridge.js
    participant API as FastAPI /api/predict
    participant Pipeline as Feature Pipeline
    participant Scaler as StandardScaler
    participant RF as RandomForestClassifier
    participant Explainer as Explain Engine

    User->>DOM: Clicks "SAMPLE FRAUD" or "EVALUATE"
    DOM->>Bridge: Triggers window.analyze()
    Note over Bridge: 1. Start scan animation (gauge pulsing)<br/>2. Clear previous drivers<br/>3. Start 650ms minimum timer
    
    par Scan Delay & Network Request
        Bridge->>API: POST /api/predict {type, amount, sb, rb, time}
        API->>Pipeline: extract_features(request)
        Pipeline-->>API: 9-feature DataFrame
        API->>Scaler: scaler.transform(df)
        Scaler-->>API: Scaled feature vector
        API->>RF: model.predict_proba(scaled_vector)
        RF-->>API: Risk probability (e.g., 0.3200)
        API->>Explainer: generate_reasons(inputs, risk, importances)
        Explainer-->>API: List of ReasonItem objects
        API-->>Bridge: 200 OK {risk: 0.32, verdict: "CLEARED", reasons: [...]}
    and UI Visual Delay
        Note over Bridge: Await 650ms scan animation
    end

    Bridge->>DOM: Stop scan animation
    Bridge->>DOM: Update gaugeArc stroke & offset
    Bridge->>DOM: Animate gaugeNum count-up (0% -> 32%)
    Bridge->>DOM: Update #verdict ("CLEARED" or "FLAGGED")
    Bridge->>DOM: Render staggered reasons (.why-row, .why-dot)
    Bridge->>DOM: Flash result effect (.flash-clear / .flash-flag)
    Note over Bridge: Cache result for synchronous #logBtn clicks
```

---

## 4. Synchronous Transaction Logging Flow

When the user clicks the **"LOG TRANSACTION"** button (`#logBtn`):

```mermaid
sequenceDiagram
    autonumber
    actor User as User
    participant LogBtn as #logBtn
    participant Bridge as bridge.js (computeRisk)
    participant Ledger as #ledgerBody (DOM)

    User->>LogBtn: Clicks "LOG TRANSACTION"
    LogBtn->>Bridge: Calls computeRisk() synchronously
    alt Cache Hit (Form matches latest API evaluation)
        Bridge-->>LogBtn: Returns cached {risk: backend_risk}
    else Cache Miss / Offline
        Bridge-->>LogBtn: Returns original client-side computed risk
    end
    LogBtn->>Ledger: Prepends new entry to ledger table
    Note over Ledger: Visual animation displays evaluated risk %
```

---

## 5. Offline Fallback Architecture

If the backend server is stopped or network connectivity fails:

1. `fetchPrediction()` in `bridge.js` catches the network error.
2. An informative log is written to `console.warn` for developer diagnostics.
3. The bridge invokes the original `_origRenderSim(true)` function stored on initialization.
4. The user observes the original client-side heuristic without broken layouts, alert dialogs, or unhandled promise rejections.
