/**
 * FRAUD SHIELD Backend Integration Bridge
 * 
 * Injected in-memory before </body> when served via FastAPI.
 * Connects the UI to the live ML backend (/api/predict) while strictly
 * preserving original animations, layout, CSS classes, and fallback behavior.
 * 
 * Rules:
 * - Read-only DOM contracts: reuse existing IDs, classes, and globals.
 * - Cache results so synchronous computeRisk() callers (#logBtn) receive backend risk.
 * - Graceful fallback to original client-side logic if backend is unreachable.
 * - Zero intrusive UI; diagnostics go to console.debug/warn only.
 */

(function () {
  console.debug("[FraudShield Bridge] Initializing integration bridge...");

  // Capture original UI functions
  const _origComputeRisk = window.computeRisk;
  const _origAnalyze = window.analyze;
  const _origRenderSim = window.renderSim;

  // Cache for the latest backend prediction
  let cachedPrediction = null;
  let activeAnalyzeToken = 0;
  let inputDebounceTimer = null;

  // Determine API base URL (supports both same-origin / and file:// origins)
  const isFileOrigin = window.location.protocol === "file:";
  const API_BASE = isFileOrigin ? "http://127.0.0.1:8000" : "";

  /**
   * Generates a cache key representing current form input state.
   */
  function getInputSignature() {
    return [
      txType ? txType.value : "",
      txAmt ? parseCurrency(txAmt.value) : 0,
      txSB ? parseCurrency(txSB.value) : 0,
      txRB ? parseCurrency(txRB.value) : 0,
      txTime ? txTime.value : ""
    ].join("|");
  }

  /**
   * Fetches prediction from the backend API.
   * Returns Promise<PredictResponse | null>
   */
  async function fetchPrediction() {
    const payload = {
      type: txType ? txType.value : "TRANSFER",
      amount: parseCurrency(txAmt ? txAmt.value : 0),
      sender_balance: parseCurrency(txSB ? txSB.value : 0),
      recipient_balance: parseCurrency(txRB ? txRB.value : 0),
      time: txTime && txTime.value ? txTime.value : "12:00"
    };

    // Skip network request if inputs violate basic limits
    if (payload.amount < 1 || payload.amount > 1000000 || payload.sender_balance < 0 || payload.recipient_balance < 0) {
      return null;
    }

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        console.warn(`[FraudShield Bridge] API error HTTP ${response.status}`);
        return null;
      }

      const data = await response.json();
      data._signature = getInputSignature();
      cachedPrediction = data;
      console.debug("[FraudShield Bridge] Real ML inference received:", data);
      return data;
    } catch (err) {
      console.warn("[FraudShield Bridge] Backend unreachable, using client heuristic:", err);
      return null;
    }
  }

  /**
   * Renders the backend prediction using the UI's exact DOM structures and CSS classes.
   */
  function renderBackendResult(pred, animated) {
    if (!pred) return;
    animated = (animated === true) && !window.REDUCED;

    const risk = pred.risk;
    const pct = Math.round(risk * 100);
    const isFlagged = pred.verdict === "FLAGGED";

    // 1. Gauge Arc & Value
    if (gaugeArc) {
      gaugeArc.style.strokeDashoffset = CIRC - CIRC * risk;
      gaugeArc.style.stroke = isFlagged ? "#EF4444" : "#D4AF37";
    }
    if (typeof setGaugeNum === "function") {
      setGaugeNum(pct, animated);
    }
    if (gaugeNum) {
      gaugeNum.setAttribute("fill", isFlagged ? "#EF4444" : "#F5E0A3");
    }

    // 2. Risk Probability Header
    const probLine = document.getElementById("probLine");
    if (probLine) {
      probLine.textContent = `RISK PROBABILITY: ${pct}%`;
    }

    // 3. Verdict
    if (verdictEl) {
      verdictEl.textContent = pred.verdict;
      verdictEl.className = isFlagged ? "verdict flagged mono" : "verdict clear mono";
    }

    // 4. Reasons and Drivers
    if (driversEl && Array.isArray(pred.reasons)) {
      const cls = ["why-low", "why-med", "why-high"];
      const flaggedCount = pred.reasons.filter(r => r.level > 0).length;
      
      const summaryText = isFlagged
        ? "Flagged because of the warning signs below."
        : (flaggedCount ? "Cleared, but a few things were worth a look." : "Cleared. Nothing unusual about this transaction.");

      const rowsHtml = pred.reasons.map((r, i) => {
        const levelCls = cls[r.level] || "why-low";
        const animCls = animated ? " why-anim" : "";
        const animDelay = animated ? ` style="animation-delay:${280 + i * 140}ms"` : "";
        return `<div class="why-row ${levelCls}${animCls}"${animDelay}><span class="why-dot"></span><span>${r.text}</span></div>`;
      }).join("");

      driversEl.innerHTML = `<div class="why-summary">${summaryText}</div>${rowsHtml}`;
    }

    // 5. Visual Flash Effect
    if (animated && typeof flashResult === "function") {
      flashResult(isFlagged);
    }
  }

  /**
   * Overridden computeRisk()
   * Intercepted to ensure synchronous callers (like #logBtn) obtain the real model's risk.
   */
  window.computeRisk = function () {
    const currentSig = getInputSignature();
    if (cachedPrediction && cachedPrediction._signature === currentSig) {
      return {
        risk: cachedPrediction.risk,
        depletionScore: 0,
        destScore: 0,
        timeScore: 0
      };
    }
    // Fall back to original client heuristic if cache is not ready
    return _origComputeRisk();
  };

  /**
   * Overridden analyze()
   * Runs the scan animation and concurrently fetches prediction from backend.
   */
  window.analyze = async function () {
    const token = ++activeAnalyzeToken;

    if (window.REDUCED) {
      const pred = await fetchPrediction();
      if (token !== activeAnalyzeToken) return;
      if (pred) renderBackendResult(pred, false);
      else _origRenderSim(false);
      return;
    }

    // Trigger existing scan animation state
    const gw = document.querySelector(".gauge-wrap");
    if (gw) {
      gw.classList.remove("scanning");
      void gw.offsetWidth;
      gw.classList.add("scanning");
    }
    if (verdictEl) {
      verdictEl.textContent = "ANALYZING…";
      verdictEl.className = "verdict scan mono";
    }
    const probLine = document.getElementById("probLine");
    if (probLine) probLine.textContent = "RISK PROBABILITY: …";
    if (driversEl) driversEl.innerHTML = "";
    if (gaugeArc) gaugeArc.style.strokeDashoffset = CIRC;
    if (typeof cancelAnimationFrame === "function" && window.gaugeRaf) {
      cancelAnimationFrame(window.gaugeRaf);
    }
    if (gaugeNum) gaugeNum.textContent = "0%";

    // Start fetch and minimum scan timer concurrently (650ms matches UI feel)
    const scanDelayPromise = new Promise(resolve => setTimeout(resolve, 650));
    const [prediction] = await Promise.all([fetchPrediction(), scanDelayPromise]);

    if (token !== activeAnalyzeToken) return;

    if (gw) gw.classList.remove("scanning");

    if (prediction) {
      renderBackendResult(prediction, true);
    } else {
      // Graceful fallback to original simulation
      _origRenderSim(true);
    }
  };

  /**
   * Overridden renderSim(animated)
   * If a cached prediction matches the current inputs, render it.
   * Otherwise, fetch in background and fall back to local heuristic for instant preview.
   */
  window.renderSim = function (animated) {
    const currentSig = getInputSignature();
    if (cachedPrediction && cachedPrediction._signature === currentSig) {
      renderBackendResult(cachedPrediction, animated);
      return;
    }

    // Instant local preview while debouncing backend fetch
    _origRenderSim(animated);

    clearTimeout(inputDebounceTimer);
    inputDebounceTimer = setTimeout(async () => {
      const pred = await fetchPrediction();
      if (pred && getInputSignature() === currentSig) {
        renderBackendResult(pred, false);
      }
    }, 250);
  };

  // Initial fetch to sync real ML prediction on initial load
  fetchPrediction().then(pred => {
    if (pred) renderBackendResult(pred, false);
  });

  console.debug("[FraudShield Bridge] Bridge successfully attached.");
})();
