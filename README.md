# Fourier Pricing of European Options (Carr-Madan FFT Method)
**BlackSwan Quants Project 2025**

## 1. Project Abstract & Goals

This project implements an option pricing engine in Python based on Fourier methods, specifically the **Carr-Madan Fast Fourier Transform (FFT) approach**.

The core idea is to value options from the **characteristic function** of the log-price. This methodology avoids the need for an explicit probability density, which is unavailable in many realistic financial models (like Heston or Merton).

The project's emphasis is **educational and methodological**, aiming for clear derivations, well-tested code, and reproducible experiments.

### Key Objectives

* **Build:** Create a reusable, modular Fourier pricer in Python.
* **Validate (Phase 1):** Derive and validate the Fourier pricing formula against the closed-form Black-Scholes solution.
* **Analyze (Phase 2):** Study the numerical stability and accuracy by sweeping key parameters (damping factor $\alpha$, frequency grid step $\eta$, number of nodes $N$).
* **Extend (Phase 3, Optional):** Extend the pricer to advanced models by "plugging in" the characteristic functions for **Merton (jump-diffusion)** and **Heston (stochastic volatility)**.

### Deliverables

* **Code:** A Python package for Carr-Madan pricing.
* **Notebooks:** Jupyter notebooks for validation vs. Black-Scholes, convergence plots, and stability analysis.
* **Paper:** A final report detailing the methodology, implementation, and results.
* **(Optional) App:** A small Streamlit demo for interactive pricing.

---

## 2. Methodology

The project will follow a structured, bottom-up approach:

1.  **Literature Review:** Consolidate theory on risk-neutral pricing, the Carr-Madan FFT method, and the characteristic functions for the Black-Scholes, Merton, and Heston models.
2.  **Derivation:** Formally derive the option pricing formula as a Fourier transform, including the role of the damping factor.
3.  **Implementation:** Code the core components in Python, including:
    * Black-Scholes characteristic function.
    * The Carr-Madan integral using the trapezoidal rule.
    * A vectorized FFT variant for speed.
4.  **Validation:** Test the implementation rigorously against the known Black-Scholes closed-form solution. We will also verify **put-call parity** and **strike monotonicity**.
5.  **Extension (Optional):** Implement the characteristic functions for Heston and Merton models.
6.  **Write-up:** Document all findings, derivations, and code in a final Overleaf (LaTeX) report.

---

## 3. Tech Stack

* **Language:** Python
* **Core Libraries:** `numpy`, `scipy` (for FFT and integration), `matplotlib` (for plots), `pandas`
* **Reporting:** Overleaf (LaTeX)
* **Optional Demo:** `streamlit`

---

## 4. Project Timeline & Milestones

The project is scheduled to run from **October 27, 2025,** to **January 17, 2026**.

| Phase | Task | Start Date | End Date |
| :--- | :--- | :--- | :--- |
| **Phase 1: Setup** | **Literature Review** | 27/10/25 | 09/11/25 |
| | *Milestone: Lit review summary ready* | *09/11/25* | *09/11/25* |
| | **Methodology Setup** | 10/11/25 | 16/11/25 |
| | *Milestone: Method spec locked* | *16/11/25* | *16/11/25* |
| **Phase 2: Core** | **Implementation** | 17/11/25 | 30/11/25 |
| | *Milestone: First pricing pass* | *30/11/25* | *30/11/25* |
| | **Experiments & Validation** | 01/12/25 | 14/12/25 |
| | *Milestone: Experiments complete* | *14/12/25* | *14/12/25* |
| **Phase 3: Ext.** | **Extensions (Optional)** | 15/12/25 | 21/12/25 |
| | *Milestone: Extensions draft* | *21/12/25* | *21/12/25* |
| **Phase 4: Final** | **Draft Writing** | 22/12/25 | 04/01/26 |
| | *Milestone: First draft ready* | *04/01/26* | *04/01/26* |
| | **Revisions & Finalization** | 05/01/26 | 17/01/26 |
| | **Final Milestone: Release v1.0** | **17/01/26** | **17/01/26** |

---

## 5. Key References

* Carr, P., and D. Madan (1999). *Option valuation using the fast fourier transform.* Journal of Computational Finance.
* Heston, S. L. (1993). *A closed-form solution for options with stochastic volatility...* The Review of Financial Studies.
* Merton, R. C. (1976). *Option pricing when underlying stock returns are discontinuous.* Journal of Financial Economics.
* Shreve, S. E. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models.* Springer.
* Jacod, J., and P. Protter (2004). *Probability Essentials.* Springer.
