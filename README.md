# Fourier Pricing of European Options (Carr-Madan FFT Method)
**BlackSwan Quants Project 2025**

## 1. Project Abstract & Goals

[cite_start]This project implements an option pricing engine in Python based on Fourier methods, specifically the **Carr-Madan Fast Fourier Transform (FFT) approach**[cite: 1, 10].

[cite_start]The core idea is to value options from the **characteristic function** of the log-price[cite: 109]. [cite_start]This methodology avoids the need for an explicit probability density, which is unavailable in many realistic financial models (like Heston or Merton)[cite: 109, 112, 127].

[cite_start]The project's emphasis is **educational and methodological**, aiming for clear derivations, well-tested code, and reproducible experiments[cite: 117].

### Key Objectives

* [cite_start]**Build:** Create a reusable, modular Fourier pricer in Python[cite: 10, 116].
* [cite_start]**Validate (Phase 1):** Derive and validate the Fourier pricing formula against the closed-form Black-Scholes solution[cite: 11, 28, 110].
* [cite_start]**Analyze (Phase 2):** Study the numerical stability and accuracy by sweeping key parameters (damping factor $\alpha$, frequency grid step $\eta$, number of nodes $N$)[cite: 18, 111, 139].
* [cite_start]**Extend (Phase 3, Optional):** Extend the pricer to advanced models by "plugging in" the characteristic functions for **Merton (jump-diffusion)** and **Heston (stochastic volatility)**[cite: 19, 112].

### Deliverables

* [cite_start]**Code:** A Python package for Carr-Madan pricing[cite: 12, 120].
* [cite_start]**Notebooks:** Jupyter notebooks for validation vs. Black-Scholes, convergence plots, and stability analysis[cite: 12, 121].
* [cite_start]**Paper:** A final report detailing the methodology, implementation, and results[cite: 12, 122].
* [cite_start]**(Optional) App:** A small Streamlit demo for interactive pricing[cite: 13, 123].

---

## 2. Methodology

[cite_start]The project will follow a structured, bottom-up approach[cite: 135]:

1.  [cite_start]**Literature Review:** Consolidate theory on risk-neutral pricing, the Carr-Madan FFT method, and the characteristic functions for the Black-Scholes, Merton, and Heston models[cite: 15, 136].
2.  [cite_start]**Derivation:** Formally derive the option pricing formula as a Fourier transform, including the role of the damping factor[cite: 16, 137].
3.  **Implementation:** Code the core components in Python, including:
    * [cite_start]Black-Scholes characteristic function[cite: 203].
    * [cite_start]The Carr-Madan integral using the trapezoidal rule[cite: 203].
    * [cite_start]A vectorized FFT variant for speed[cite: 138, 203].
4.  [cite_start]**Validation:** Test the implementation rigorously against the known Black-Scholes closed-form solution[cite: 139, 203]. [cite_start]We will also verify **put-call parity** and **strike monotonicity**[cite: 18, 139, 203].
5.  [cite_start]**Extension (Optional):** Implement the characteristic functions for Heston and Merton models[cite: 19, 140, 203].
6.  [cite_start]**Write-up:** Document all findings, derivations, and code in a final Overleaf (LaTeX) report[cite: 20, 140].

---

## 3. Tech Stack

* [cite_start]**Language:** Python [cite: 156]
* [cite_start]**Core Libraries:** `numpy`, `scipy` (for FFT and integration), `matplotlib` (for plots), `pandas` [cite: 158]
* [cite_start]**Reporting:** Overleaf (LaTeX) [cite: 160]
* [cite_start]**Optional Demo:** `streamlit` [cite: 158]

---

## 4. Project Timeline & Milestones

[cite_start]The project is scheduled to run from **September 29, 2025,** to **December 21, 2025**[cite: 36].

| Phase | Task | Start Date | End Date |
| :--- | :--- | :--- | :--- |
| **Phase 1: Setup** | **Literature Review** | 29/09/25 | 12/10/25 |
| | *Milestone: Lit review summary ready* | *12/10/25* | *12/10/25* |
| | **Methodology Setup** | 13/10/25 | 19/10/25 |
| | *Milestone: Method spec locked* | *19/10/25* | *19/10/25* |
| **Phase 2: Core** | **Implementation** | 20/10/25 | 02/11/25 |
| | *Milestone: First pricing pass* | *02/11/25* | *02/11/25* |
| | **Experiments & Validation** | 03/11/25 | 16/11/25 |
| | *Milestone: Experiments complete* | *16/11/25* | *16/11/25* |
| **Phase 3: Ext.** | **Extensions (Optional)** | 17/11/25 | 23/11/25 |
| | *Milestone: Extensions draft* | *23/11/25* | *23/11/25* |
| **Phase 4: Final** | **Draft Writing** | 24/11/25 | 07/12/25 |
| | *Milestone: First draft ready* | *07/12/25* | *07/12/25* |
| | **Revisions & Finalization** | 08/12/25 | 20/12/25 |
| | **Final Milestone: Release v1.0** | **20/12/25** | **20/12/25** |

[cite_start]*(Source for all timeline data: [cite: 40, 45, 48, 203, 206])*

---

## 5. Key References

* Carr, P., and D. Madan (1999). [cite_start]*Option valuation using the fast fourier transform.* Journal of Computational Finance[cite: 221].
* Heston, S. L. (1993). [cite_start]*A closed-form solution for options with stochastic volatility...* The Review of Financial Studies[cite: 222].
* Merton, R. C. (1976). [cite_start]*Option pricing when underlying stock returns are discontinuous.* Journal of Financial Economics[cite: 225].
* Shreve, S. E. (2004). [cite_start]*Stochastic Calculus for Finance II: Continuous-Time Models.* Springer[cite: 228].
* Jacod, J., and P. Protter (2004). [cite_start]*Probability Essentials.* Springer[cite: 224].
