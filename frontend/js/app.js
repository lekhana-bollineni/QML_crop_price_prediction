/**
 * Quantum-Enhanced Crop Market Price Prediction Dashboard
 * Client-Side Application Logic with Dynamic API Integration and Chart.js
 */

const API_BASE = "";

// State management
let statesData = [];
let modelMetrics = [];
let comparisonData = {};
let historicalChartInstance = null;
let charts = {};

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  loadInitialData();
  setupPredictionForm();
  setupFilters();
  setupLightbox();
});

// =============================================================================
// NAVIGATION & TABS
// =============================================================================
function initNavigation() {
  const tabs = document.querySelectorAll(".nav-tab-btn");
  const sections = document.querySelectorAll(".tab-section");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      tabs.forEach(t => t.classList.remove("active"));
      sections.forEach(s => s.classList.remove("active"));

      tab.classList.add("active");
      const targetSection = document.getElementById(targetId);
      if (targetSection) {
        targetSection.classList.add("active");
        if (targetId === "charts") {
          renderAllCharts();
        }
      }
    });
  });
}

// =============================================================================
// INITIAL DATA LOADING
// =============================================================================
async function loadInitialData() {
  try {
    // 1. Fetch States
    const resStates = await fetch(`${API_BASE}/api/states`);
    const statesJson = await resStates.json();
    if (statesJson.status === "success") {
      statesData = statesJson.states;
      populateStates(statesData);
    }

    // 2. Fetch Comparison & Metrics
    const resComp = await fetch(`${API_BASE}/api/comparison`);
    comparisonData = await resComp.json();
    if (comparisonData.status === "success") {
      modelMetrics = comparisonData.metrics || [];
      renderComparisonTable(modelMetrics, comparisonData.leaders || {});
      renderComputationalTable(comparisonData.computational || []);
      renderObjectiveLeaders(comparisonData.leaders || {});
    }

    // Pre-select default location: Maharashtra -> Nashik -> Nasik
    const stateSelect = document.getElementById("select-state");
    if (stateSelect && statesData.includes("Maharashtra")) {
      stateSelect.value = "Maharashtra";
      await onStateChange("Maharashtra", "Nashik", "Nasik");
    }
  } catch (err) {
    console.error("Error loading initial dashboard data:", err);
  }
}

// =============================================================================
// DYNAMIC DROPDOWNS: STATE -> DISTRICT -> MARKET
// =============================================================================
function populateStates(states) {
  const stateSelect = document.getElementById("select-state");
  if (!stateSelect) return;
  stateSelect.innerHTML = `<option value="">-- Select State --</option>` + 
    states.map(s => `<option value="${s}">${s}</option>`).join("");

  stateSelect.addEventListener("change", async (e) => {
    const st = e.target.value;
    await onStateChange(st);
  });
}

async function onStateChange(state, defaultDistrict = null, defaultMarket = null) {
  const districtSelect = document.getElementById("select-district");
  const marketSelect = document.getElementById("select-market");
  districtSelect.innerHTML = `<option value="">Loading districts...</option>`;
  marketSelect.innerHTML = `<option value="">-- Select District First --</option>`;

  if (!state) {
    districtSelect.innerHTML = `<option value="">-- Select State First --</option>`;
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/districts?state=${encodeURIComponent(state)}`);
    const json = await res.json();
    if (json.status === "success") {
      districtSelect.innerHTML = `<option value="">-- Select District --</option>` +
        json.districts.map(d => `<option value="${d}">${d}</option>`).join("");

      if (defaultDistrict && json.districts.includes(defaultDistrict)) {
        districtSelect.value = defaultDistrict;
        await onDistrictChange(state, defaultDistrict, defaultMarket);
      }
    }
  } catch (err) {
    console.error("Failed to load districts:", err);
  }

  districtSelect.onchange = async () => {
    await onDistrictChange(state, districtSelect.value);
  };
}

async function onDistrictChange(state, district, defaultMarket = null) {
  const marketSelect = document.getElementById("select-market");
  marketSelect.innerHTML = `<option value="">Loading markets...</option>`;

  if (!district) {
    marketSelect.innerHTML = `<option value="">-- Select District First --</option>`;
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/markets?state=${encodeURIComponent(state)}&district=${encodeURIComponent(district)}`);
    const json = await res.json();
    if (json.status === "success") {
      marketSelect.innerHTML = `<option value="">-- Select Market --</option>` +
        json.markets.map(m => `<option value="${m}">${m}</option>`).join("");

      if (defaultMarket && json.markets.includes(defaultMarket)) {
        marketSelect.value = defaultMarket;
        loadHistoricalTrend(state, district, defaultMarket);
      } else if (json.markets.length > 0) {
        marketSelect.value = json.markets[0];
        loadHistoricalTrend(state, district, json.markets[0]);
      }
    }
  } catch (err) {
    console.error("Failed to load markets:", err);
  }

  marketSelect.onchange = () => {
    loadHistoricalTrend(state, district, marketSelect.value);
  };
}

// =============================================================================
// HISTORICAL TREND FOR SELECTED MANDI
// =============================================================================
async function loadHistoricalTrend(state, district, market) {
  if (!state || !district || !market) return;
  try {
    const res = await fetch(`${API_BASE}/api/historical-prices?state=${encodeURIComponent(state)}&district=${encodeURIComponent(district)}&market=${encodeURIComponent(market)}`);
    const json = await res.json();
    if (json.status === "success" && json.series.length > 0) {
      renderHistoricalPriceChart(json.series, market);
    }
  } catch (err) {
    console.error("Error loading historical prices:", err);
  }
}

function renderHistoricalPriceChart(series, marketName) {
  const ctx = document.getElementById("chart-mandi-history");
  if (!ctx) return;

  const labels = series.map(d => d.date);
  const prices = series.map(d => d.price);
  const arrivals = series.map(d => d.arrivals);

  if (historicalChartInstance) {
    historicalChartInstance.destroy();
  }

  historicalChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: `${marketName} Modal Price (₹/qtl)`,
          data: prices,
          borderColor: "#2a9d8f",
          backgroundColor: "rgba(42, 157, 143, 0.15)",
          fill: true,
          tension: 0.3,
          borderWidth: 2.5,
          yAxisID: "y"
        },
        {
          label: `Arrivals (Tonnes)`,
          data: arrivals,
          borderColor: "#e76f51",
          backgroundColor: "rgba(231, 111, 81, 0.1)",
          fill: false,
          tension: 0.2,
          borderWidth: 1.5,
          borderDash: [5, 5],
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#f0f4f8" } },
        tooltip: { mode: "index", intersect: false }
      },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: {
          type: "linear",
          position: "left",
          title: { display: true, text: "₹ / Quintal", color: "#2a9d8f" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.05)" }
        },
        y1: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Tonnes", color: "#e76f51" },
          ticks: { color: "#94a3b8" },
          grid: { drawOnChartArea: false }
        }
      }
    }
  });
}

// =============================================================================
// PREDICTION & REVENUE CALCULATION
// =============================================================================
function setupPredictionForm() {
  const form = document.getElementById("prediction-form");
  const btnPredict = document.getElementById("btn-predict");
  const quantityInput = document.getElementById("input-quantity");

  // Quantity input real-time listener
  if (quantityInput) {
    quantityInput.addEventListener("input", () => {
      recalculateRevenue();
    });
  }

  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const state = document.getElementById("select-state").value;
    const district = document.getElementById("select-district").value;
    const market = document.getElementById("select-market").value;
    const model = document.getElementById("select-model").value;
    const month = parseInt(document.getElementById("select-month").value);
    const year = parseInt(document.getElementById("select-year").value);
    const quantity = parseFloat(document.getElementById("input-quantity").value) || null;

    if (!state || !district || !market) {
      alert("Please select a State, District, and Market.");
      return;
    }

    btnPredict.disabled = true;
    btnPredict.innerHTML = `<span>Simulating Quantum Circuit & Computing...</span>`;

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, district, market, model, month, year, quantity })
      });

      const data = await response.json();
      if (response.ok && data.status === "success") {
        displayPredictionResult(data);
      } else {
        alert(`Prediction Error: ${data.error || "Unable to generate prediction."}`);
      }
    } catch (err) {
      console.error("Prediction request failed:", err);
      alert("Failed to communicate with prediction backend API.");
    } finally {
      btnPredict.disabled = false;
      btnPredict.innerHTML = `<span>PREDICT PRICE</span>`;
    }
  });
}

let lastPredictedPrice = 0;

function displayPredictionResult(res) {
  lastPredictedPrice = res.predicted_price;

  document.getElementById("res-price").innerText = res.predicted_price.toLocaleString("en-IN", { minimumFractionDigits: 2 });
  document.getElementById("res-model-badge").innerText = res.model;
  document.getElementById("res-model-badge").className = `badge ${res.model.includes('Q') || res.model === 'VQR' ? 'badge-quantum' : 'badge-classical'}`;
  
  document.getElementById("res-market").innerText = res.market;
  document.getElementById("res-date").innerText = `${res.month_name} ${res.year}`;
  document.getElementById("res-latency").innerText = `${(res.inference_latency_seconds * 1000).toFixed(1)} ms`;
  document.getElementById("res-historical-price").innerText = res.latest_historical_mandi_price ? `₹ ${res.latest_historical_mandi_price.toFixed(2)}` : 'N/A';

  const revCard = document.getElementById("revenue-container");
  if (res.estimated_revenue_rs !== undefined && res.estimated_revenue_rs !== null) {
    revCard.style.display = "block";
    document.getElementById("res-revenue").innerText = `₹ ${res.estimated_revenue_rs.toLocaleString("en-IN", { minimumFractionDigits: 2 })}`;
    document.getElementById("res-quantity-display").innerText = `${res.quantity_quintals} Quintals`;
  } else {
    revCard.style.display = "none";
  }
}

function recalculateRevenue() {
  const qty = parseFloat(document.getElementById("input-quantity").value);
  const revCard = document.getElementById("revenue-container");
  if (!isNaN(qty) && qty > 0 && lastPredictedPrice > 0) {
    revCard.style.display = "block";
    const totalRev = Math.round(qty * lastPredictedPrice * 100) / 100;
    document.getElementById("res-revenue").innerText = `₹ ${totalRev.toLocaleString("en-IN", { minimumFractionDigits: 2 })}`;
    document.getElementById("res-quantity-display").innerText = `${qty} Quintals`;
  }
}

// =============================================================================
// COMPARISON TABLES & OBJECTIVE LEADERS (DYNAMIC)
// =============================================================================
function renderComparisonTable(metrics, leaders) {
  const tbody = document.getElementById("table-metrics-body");
  const statTbody = document.getElementById("table-statistical-body");

  if (tbody) {
    tbody.innerHTML = metrics.map(m => {
      const isBestMAE = m.model === leaders.best_by_MAE;
      const isBestRMSE = m.model === leaders.best_by_RMSE;
      const isBestR2 = m.model === leaders.best_by_R2;
      const isBestMAPE = m.model === leaders.best_by_MAPE;
      const isQuantum = m.type === "Quantum";
      const stdVal = m.error_std_dev !== undefined ? `₹${Number(m.error_std_dev).toFixed(2)}` : 'N/A';

      return `
        <tr class="${isBestMAE ? 'highlight-best' : ''}">
          <td><strong>${m.model}</strong></td>
          <td><span class="badge ${isQuantum ? 'badge-quantum' : 'badge-classical'}">${m.type}</span></td>
          <td class="${isBestMAE ? 'metric-pill' : ''}">₹${Number(m.MAE).toFixed(2)} ${isBestMAE ? '🏆' : ''}</td>
          <td class="${isBestRMSE ? 'metric-pill' : ''}">₹${Number(m.RMSE).toFixed(2)}</td>
          <td class="${isBestR2 ? 'metric-pill' : ''}">${Number(m.R2).toFixed(4)}</td>
          <td class="${isBestMAPE ? 'metric-pill' : ''}">${Number(m.MAPE).toFixed(2)}%</td>
          <td>${stdVal}</td>
          <td>${Number(m.training_time_seconds).toFixed(2)}s</td>
          <td>${Number(m.prediction_time_seconds).toFixed(2)}s</td>
        </tr>
      `;
    }).join("");
  }

  if (statTbody) {
    statTbody.innerHTML = metrics.map(m => {
      const isQuantum = m.type === "Quantum";
      const isRef = m.model === "Random Forest" || m.paired_t_statistic === 0 || m.t_statistic === 0;
      const tStat = m.t_statistic !== undefined ? Number(m.t_statistic).toFixed(4) : (m.paired_t_statistic || '0.0000');
      const pValT = m.p_value_paired_ttest || m.paired_t_p_value || (isRef ? '1.0000' : 'N/A');
      const fStat = m.f_statistic !== undefined ? Number(m.f_statistic).toFixed(4) : '1.0000';
      const pValF = m.f_test_p_value || (isRef ? '1.0000' : 'N/A');
      const stdDev = m.error_std_dev !== undefined ? `₹${Number(m.error_std_dev).toFixed(2)}` : 'N/A';
      const optimizer = m.optimizer || "Optimized via Validation Tuning";
      const tConclusion = isRef ? "Reference Benchmark" : (parseFloat(pValT) < 0.05 ? "<span style='color:#ef476f;font-weight:600;'>Significant (p < 0.05)</span>" : "<span style='color:#38b000;font-weight:600;'>Not Significant</span>");

      return `
        <tr>
          <td><strong>${m.model}</strong></td>
          <td><span class="badge ${isQuantum ? 'badge-quantum' : 'badge-classical'}">${m.type}</span></td>
          <td>${stdDev}</td>
          <td><code>${tStat}</code></td>
          <td><code>${pValT}</code></td>
          <td>${tConclusion}</td>
          <td><code>${fStat}</code></td>
          <td><code>${pValF}</code></td>
          <td style="font-size:0.8rem; color:var(--text-muted);">${optimizer}</td>
        </tr>
      `;
    }).join("");
  }
}

function renderComputationalTable(comp) {
  const tbody = document.getElementById("table-computational-body");
  if (!tbody) return;

  tbody.innerHTML = comp.map(c => `
    <tr>
      <td><strong>${c.model}</strong></td>
      <td><span class="badge ${c.type === 'Quantum' ? 'badge-quantum' : 'badge-classical'}">${c.type}</span></td>
      <td>${c.training_samples.toLocaleString()}</td>
      <td>${c.test_samples.toLocaleString()}</td>
      <td><strong>${c.training_time_seconds.toFixed(2)}s</strong></td>
      <td>${c.prediction_time_seconds.toFixed(2)}s</td>
    </tr>
  `).join("");
}

function renderObjectiveLeaders(leaders) {
  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.innerText = val || "N/A";
  };
  setEl("leader-mae", leaders.best_by_MAE);
  setEl("leader-rmse", leaders.best_by_RMSE);
  setEl("leader-r2", leaders.best_by_R2);
  setEl("leader-mape", leaders.best_by_MAPE);
  setEl("leader-fast-train", leaders.fastest_training_model);
  setEl("leader-fast-infer", leaders.fastest_prediction_model);
}

// =============================================================================
// CHART.JS INTERACTIVE DASHBOARD SUITE
// =============================================================================
function renderAllCharts() {
  if (!modelMetrics || modelMetrics.length === 0) return;

  const models = modelMetrics.map(m => m.model);
  const colors = {
    "Random Forest": "#2E7D32",
    "XGBoost": "#1565C0",
    "LightGBM": "#00838F",
    "VQR": "#E65100",
    "QNN": "#6A1B9A",
    "QSVR": "#AD1457"
  };
  const barColors = models.map(m => colors[m] || "#2a9d8f");

  // Helper chart builder
  const createBarChart = (canvasId, label, data, yTitle, isCurrency = false, isPercent = false) => {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (charts[canvasId]) charts[canvasId].destroy();

    charts[canvasId] = new Chart(ctx, {
      type: "bar",
      data: {
        labels: models,
        datasets: [{
          label: label,
          data: data,
          backgroundColor: barColors,
          borderColor: "#222",
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => `${label}: ${isCurrency ? '₹' : ''}${item.raw}${isPercent ? '%' : ''}`
            }
          }
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
          y: {
            title: { display: true, text: yTitle, color: "#94a3b8" },
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.06)" }
          }
        }
      }
    });
  };

  // 1. MAE
  createBarChart("chart-mae", "MAE (₹/qtl)", modelMetrics.map(m => m.MAE), "Error Scale (Lower is Better)", true);
  // 2. RMSE
  createBarChart("chart-rmse", "RMSE (₹/qtl)", modelMetrics.map(m => m.RMSE), "RMSE Scale (Lower is Better)", true);
  // 3. R2
  createBarChart("chart-r2", "R2 Score", modelMetrics.map(m => m.R2), "Explained Variance (Higher is Better)");
  // 4. MAPE
  createBarChart("chart-mape", "MAPE (%)", modelMetrics.map(m => m.MAPE), "Percentage Error", false, true);
  // 5. Training Time
  createBarChart("chart-train-time", "Training Time (s)", modelMetrics.map(m => m.training_time_seconds), "Seconds");
  // 6. Prediction Time
  createBarChart("chart-pred-time", "Prediction Latency (s)", modelMetrics.map(m => m.prediction_time_seconds), "Seconds for 5,109 samples");

  // 7. Classical vs Quantum Computational Time
  const ctxComp = document.getElementById("chart-comp-time");
  if (ctxComp) {
    if (charts["chart-comp-time"]) charts["chart-comp-time"].destroy();
    charts["chart-comp-time"] = new Chart(ctxComp, {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "Training Time (s)",
            data: modelMetrics.map(m => m.training_time_seconds),
            backgroundColor: "#1976D2",
            borderRadius: 6
          },
          {
            label: "Inference Time (s)",
            data: modelMetrics.map(m => m.prediction_time_seconds),
            backgroundColor: "#F57C00",
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#f0f4f8" } }
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { display: false } },
          y: {
            title: { display: true, text: "Wall-Clock Seconds", color: "#94a3b8" },
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.06)" }
          }
        }
      }
    });
  }

  // 8. Load predictions sample for timeline & actual vs predicted
  loadPredictionsForVisualizations();
}

async function loadPredictionsForVisualizations() {
  try {
    const res = await fetch(`${API_BASE}/api/predictions?limit=600`);
    const json = await res.json();
    if (json.status !== "success" || !json.predictions) return;

    const preds = json.predictions;
    renderMonthlyTrajectoryChart(preds);
  } catch (err) {
    console.error("Error loading sample predictions:", err);
  }
}

function renderMonthlyTrajectoryChart(preds) {
  const ctx = document.getElementById("chart-model-trajectory");
  if (!ctx) return;
  if (charts["chart-model-trajectory"]) charts["chart-model-trajectory"].destroy();

  // Aggregate monthly averages per model
  const monthlyData = {};
  const actuals = {};

  preds.forEach(p => {
    const d = p.target_date || p.date;
    if (!monthlyData[d]) monthlyData[d] = {};
    if (!monthlyData[d][p.model]) monthlyData[d][p.model] = [];
    monthlyData[d][p.model].push(p.predicted_price);
    
    if (!actuals[d]) actuals[d] = [];
    actuals[d].push(p.actual_price);
  });

  const dates = Object.keys(monthlyData).sort();
  const actualAverages = dates.map(d => {
    const arr = actuals[d] || [];
    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
  });

  const modelsList = ["Random Forest", "XGBoost", "LightGBM", "QSVR", "QNN", "VQR"];
  const colors = {
    "Random Forest": "#2E7D32",
    "XGBoost": "#1565C0",
    "LightGBM": "#00838F",
    "VQR": "#E65100",
    "QNN": "#6A1B9A",
    "QSVR": "#AD1457"
  };

  const datasets = [
    {
      label: "Actual Market Price",
      data: actualAverages,
      borderColor: "#ffffff",
      backgroundColor: "#ffffff",
      borderWidth: 3,
      tension: 0.1,
      pointRadius: 4
    }
  ];

  modelsList.forEach(m => {
    const vals = dates.map(d => {
      const arr = (monthlyData[d] && monthlyData[d][m]) || [];
      return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
    });
    datasets.push({
      label: m,
      data: vals,
      borderColor: colors[m] || "#2a9d8f",
      borderWidth: 1.8,
      tension: 0.2,
      pointRadius: 3
    });
  });

  charts["chart-model-trajectory"] = new Chart(ctx, {
    type: "line",
    data: { labels: dates, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#f0f4f8" } },
        tooltip: { mode: "index", intersect: false }
      },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.06)" } },
        y: {
          title: { display: true, text: "Average Price (₹/qtl)", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(255,255,255,0.06)" }
        }
      }
    }
  });
}

// =============================================================================
// FILTERS & SEARCH
// =============================================================================
function setupFilters() {
  const modelFilter = document.getElementById("filter-model");
  if (modelFilter) {
    modelFilter.addEventListener("change", async (e) => {
      const selected = e.target.value;
      const res = await fetch(`${API_BASE}/api/predictions?model=${encodeURIComponent(selected)}&limit=200`);
      const json = await res.json();
      if (json.status === "success") {
        renderFilteredPredictionsTable(json.predictions);
      }
    });
  }
}

function renderFilteredPredictionsTable(preds) {
  const tbody = document.getElementById("table-predictions-sample-body");
  if (!tbody || !preds) return;
  tbody.innerHTML = preds.slice(0, 50).map(p => `
    <tr>
      <td>${p.target_date || p.date}</td>
      <td>${p.state}</td>
      <td>${p.market}</td>
      <td><strong>${p.model}</strong></td>
      <td>₹${p.actual_price.toFixed(2)}</td>
      <td>₹${p.predicted_price.toFixed(2)}</td>
      <td style="color: ${p.error >= 0 ? '#ef476f' : '#38b000'}">${p.error >= 0 ? '+' : ''}${p.error.toFixed(2)}</td>
      <td>${p.percentage_error.toFixed(2)}%</td>
    </tr>
  `).join("");
}

// =============================================================================
// PUBLICATION FIGURES LIGHTBOX MODAL
// =============================================================================
function setupLightbox() {
  const overlay = document.getElementById("modal-lightbox");
  const modalImg = document.getElementById("modal-img");
  const closeBtn = document.getElementById("modal-close");

  document.querySelectorAll(".viz-item img").forEach(img => {
    img.addEventListener("click", () => {
      modalImg.src = img.src;
      overlay.classList.add("active");
    });
  });

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      overlay.classList.remove("active");
    });
  }

  if (overlay) {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) {
        overlay.classList.remove("active");
      }
    });
  }
}
