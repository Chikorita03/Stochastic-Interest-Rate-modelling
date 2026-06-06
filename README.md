# Stochastic Interest Rate Modelling

Implementation, calibration, and extension of the **Cox-Ingersoll-Ross (CIR)** short-rate model for yield curve reconstruction from a single observable input.

**Finance Club, IIT Roorkee — Open Projects 2026**

---

## Problem Statement

Given only the **3-Month Treasury yield** on any test day as a proxy for the instantaneous short rate $r_t$, reconstruct the full yield curve across maturities: **6M, 9M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y**.

Official metric: **Global Flattened Out-of-Sample R² > 0.85**

**Result achieved: R² = 0.9265 ✓**

---

## Repository Structure

```
├── data/
│   └── raw/
│       ├── train_data.csv          # Daily Treasury yields (training period)
│       └── test_data.csv           # Daily Treasury yields (test period, up to 2Y)
├── notebooks/
│   └── final_submission.ipynb      # Full pipeline notebook
├── requirements.txt
└── README.md
```

---

## Model Pipeline

| Step | Section | Method |
|---|---|---|
| Data Engineering | §3–5 | Forward-fill, interpolation, rolling z-score outlier detection |
| CIR Calibration | §6–8 | OLS (WLS transform) + QMLE multi-start (8 random starts) |
| CIR Simulation | §9 | Milstein discretisation scheme |
| Yield Reconstruction | §10–11 | Hybrid CIR features + Polynomial Ridge Regression |
| OOS Evaluation | §12 | Global flattened R² (official metric, threshold > 0.85) |
| Extension | §13 | CIR++ deterministic shift (Brigo-Mercurio, 2001) |
| Analysis | §14–15 | Three-model comparison, key questions answered |

---

## CIR Model

The CIR stochastic differential equation:

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

Zero-coupon bond price closed form:

$$P(t,T) = A(\tau)\,e^{-B(\tau)\,r_t}, \qquad \tau = T - t$$

The **square-root diffusion** $\sigma\sqrt{r_t}$ prevents negative rates when the Feller condition $2\kappa\theta \geq \sigma^2$ holds.

---

## Results

| Model | Global OOS R² |
|---|---|
| Base CIR | 0.807 |
| CIR++ (Brigo-Mercurio shift) | 0.920 |
| **Polynomial Ridge (submitted)** | **0.9265 ✓** |

---

## Setup

```bash
pip install -r requirements.txt
```

**Core dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

Open `notebooks/final_submission.ipynb` in Jupyter and run all cells. The notebook auto-locates `data/raw/` relative to its position.

---

## Key Design Choices

- **Single observable input**: only the 3M yield is used at test time — no look-ahead bias.
- **OLS over QMLE for downstream use**: κ and θ are weakly identified individually on daily data; OLS produces a more stable κ (≈ 0.25, half-life ≈ 2.7 years) used in feature engineering.
- **Per-maturity models**: one Polynomial Ridge model per maturity allows maturity-specific regularisation.
- **Spread formulation**: models predict `yield − short_rate`, reducing the regression target variance.
- **CIR++ extension**: a fixed 8-number shift vector φ(τ) calibrated on the last training date corrects systematic level bias with zero test-time parameters.

---# Stochastic-Interest-Rate-modelling
Implementation, calibration, and extension of the Cox-Ingersoll-Ross (CIR) interest rate model for yield curve reconstruction and prediction.
