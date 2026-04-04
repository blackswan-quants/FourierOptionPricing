#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace py = pybind11;
using Complex = std::complex<double>;
const double PI = 3.14159265358979323846;

// ── FFT Algorithm ───────────────────────────────────────────────────────────
void fft(std::vector<Complex>& x) {
    size_t N = x.size(); if (N <= 1) return;
    size_t j = 0;
    for (size_t i = 1; i < N; ++i) {
        size_t bit = N >> 1; while (j & bit) { j ^= bit; bit >>= 1; } j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = -2 * PI / len; Complex wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < N; i += len) {
            Complex w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                Complex u = x[i + k], v = x[i + k + len / 2] * w;
                x[i + k] = u + v; x[i + k + len / 2] = u - v; w *= wlen;
            }
        }
    }
}

// ── Characteristic Functions (S0=1 normalization for stability) ─────────────
Complex cf_bs(Complex u, double r, double q, double sigma, double T) {
    double mu = (r - q - 0.5 * sigma * sigma) * T;
    return std::exp(Complex(0.0, 1.0) * u * mu - 0.5 * sigma * sigma * u * u * T);
}
Complex cf_heston(Complex u, double r, double q, double T, double kappa, double theta, double sigma_v, double rho, double v0) {
    Complex i(0.0, 1.0); double a = kappa * theta;
    Complex d = std::sqrt(std::pow(rho * sigma_v * i * u - kappa, 2) + sigma_v * sigma_v * (i * u + u * u));
    Complex b = kappa - rho * sigma_v * i * u, g = (b - d) / (b + d), e_dT = std::exp(-d * T);
    Complex C = (i * u * (r - q) * T + a / (sigma_v * sigma_v) * ((b - d) * T - 2.0 * std::log((1.0 - g * e_dT) / (1.0 - g))));
    Complex D = (b - d) / (sigma_v * sigma_v) * ((1.0 - e_dT) / (1.0 - g * e_dT));
    return std::exp(C + D * v0);
}
Complex cf_merton(Complex u, double r, double q, double sigma, double T, double lam, double mu_j, double sig_j) {
    Complex i(0.0, 1.0); double kap = std::exp(mu_j + 0.5 * sig_j * sig_j) - 1.0;
    double drift = (r - q - 0.5 * sigma * sigma - lam * kap) * T;
    return std::exp(i * u * drift - 0.5 * sigma * sigma * u * u * T) * std::exp(lam * T * (std::exp(i * u * mu_j - 0.5 * sig_j * sig_j * u * u) - 1.0));
}
Complex cf_vg(Complex u, double r, double q, double T, double sigma, double nu, double theta_vg) {
    Complex i(0.0, 1.0); double ca = 1.0 - theta_vg * nu - 0.5 * sigma * sigma * nu;
    if (ca <= 0) return Complex(0.0, 0.0);
    double om = (1.0 / nu) * std::log(ca);
    return std::exp(i * u * (r - q + om) * T) * std::pow(1.0 - i * u * theta_vg * nu + 0.5 * sigma * sigma * nu * u * u, -T / nu);
}

// ── Robust IV Inversion (OTM switching) ─────────────────────────────────────
double bs_price_otm(double S, double K, double T, double r, double q, double sigma) {
    double s_sqrt_t = sigma * std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / s_sqrt_t;
    double d2 = d1 - s_sqrt_t;
    double q_disc = std::exp(-q * T), r_disc = std::exp(-r * T);
    if (K >= S) return S * q_disc * 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0))) - K * r_disc * 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
    else return K * r_disc * 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0))) - S * q_disc * 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
}

double compute_iv_robust(double call_price, double S, double K, double T, double r, double q) {
    double q_disc = std::exp(-q * T), r_disc = std::exp(-r * T);
    double target = (K < S) ? (call_price - S * q_disc + K * r_disc) : call_price;
    if (target <= 1e-8) return 1e-6;
    double vol = 0.4;
    for (int i = 0; i < 40; ++i) {
        double p = bs_price_otm(S, K, T, r, q, vol);
        double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        double vega = S * q_disc * std::sqrt(T) * (1.0 / std::sqrt(2.0 * PI)) * std::exp(-0.5 * d1 * d1);
        if (std::abs(p - target) < 1e-8) return vol;
        if (vega < 1e-10) break;
        vol -= (p - target) / vega;
        if (vol <= 0) vol = 1e-6;
    }
    return vol;
}

// ── Core Engine with Euler Delta ────────────────────────────────────────────
py::tuple pricing_core_extended(const std::function<Complex(Complex)>& cf, double S0, double r, double q, double T, double alpha, int N, double eta, std::string type) {
    double lambd = 2 * PI / (N * eta), b = 0.5 * N * lambd, disc = std::exp(-r * T);
    std::vector<Complex> buf(N);
    for (int j = 0; j < N; ++j) {
        double v = j * eta; Complex u = v - Complex(0.0, alpha + 1.0);
        double weight = (eta / 3.0) * (j == 0 ? 1.0 : (j % 2 == 0 ? 2.0 : 4.0));
        buf[j] = std::exp(Complex(0.0, b * v)) * (disc * cf(u) / (alpha * alpha + alpha - v * v + Complex(0.0, 2 * alpha + 1) * v)) * weight;
    }
    fft(buf);
    py::array_t<double> K_arr(N), V_arr(N), D_arr(N); 
    auto pK = K_arr.mutable_unchecked<1>(), pV = V_arr.mutable_unchecked<1>(), pD = D_arr.mutable_unchecked<1>();
    double q_disc = std::exp(-q * T);
    
    // Euler Delta: c(k) - partial c(k) / partial k
    for (int m = 0; m < N; ++m) {
        double k = -b + m * lambd; pK(m) = S0 * std::exp(k);
        double ck = std::exp(-alpha * k) * buf[m].real() / PI;
        double p_m = (m > 0) ? std::exp(-alpha * (-b + (m-1)*lambd)) * buf[m-1].real() / PI : ck;
        double p_p = (m < N-1) ? std::exp(-alpha * (-b + (m+1)*lambd)) * buf[m+1].real() / PI : ck;
        double dck_dk = (p_p - p_m) / (2.0 * lambd);
        pD(m) = ck - dck_dk; // Euler Delta for normalized price
        pV(m) = S0 * ((type == "put") ? (ck - q_disc + std::exp(k) * disc) : ck);
    }
    return py::make_tuple(K_arr, V_arr, D_arr);
}

double loss_normalized_robust(const std::function<Complex(Complex)>& cf, double S0, double r, double q, double T, const std::vector<double>& strikes, const std::vector<double>& ivs) {
    double eta = (T < 0.1) ? 0.5 : 0.25; int N = (T < 0.1) ? 8192 : 4096;
    double alpha = 1.5, lambd = 2 * PI / (N * eta), b = 0.5 * N * lambd, disc = std::exp(-r * T);
    std::vector<Complex> buf(N);
    for (int j = 0; j < N; ++j) {
        double v = j * eta; Complex u = v - Complex(0.0, alpha + 1.0);
        double weight = (eta / 3.0) * (j == 0 ? 1.0 : (j % 2 == 0 ? 2.0 : 4.0));
        buf[j] = std::exp(Complex(0.0, b * v)) * (disc * cf(u) / (alpha * alpha + alpha - v * v + Complex(0.0, 2 * alpha + 1) * v)) * weight;
    }
    fft(buf);
    double err = 0;
    for (size_t i = 0; i < strikes.size(); ++i) {
        double mf = (std::log(strikes[i] / S0) + b) / lambd; int m = (int)std::floor(mf);
        if (m < 0 || m >= N - 1) { err += 1000.0; continue; }
        double p_m = std::exp(-alpha * (-b + m * lambd)) * buf[m].real() / PI, p_m1 = std::exp(-alpha * (-b + (m + 1) * lambd)) * buf[m+1].real() / PI;
        double price = S0 * (p_m + (p_m1 - p_m) * (mf - m));
        err += std::pow(compute_iv_robust(price, S0, strikes[i], T, r, q) * 100.0 - ivs[i], 2);
    }
    return err / strikes.size();
}

PYBIND11_MODULE(cpp_pricer, m) {
    m.def("loss_bs", [](double s, double S0, double r, double q, double T, std::vector<double> sk, std::vector<double> iv) { return loss_normalized_robust([&](Complex u){return cf_bs(u,r,q,s,T);}, S0, r, q, T, sk, iv); });
    m.def("loss_merton", [](double s, double l, double mj, double sj, double S0, double r, double q, double T, std::vector<double> sk, std::vector<double> iv) { return loss_normalized_robust([&](Complex u){return cf_merton(u,r,q,s,T,l,mj,sj);}, S0, r, q, T, sk, iv); });
    m.def("loss_vg", [](double s, double n, double tv, double S0, double r, double q, double T, std::vector<double> sk, std::vector<double> iv) { return loss_normalized_robust([&](Complex u){return cf_vg(u,r,q,T,s,n,tv);}, S0, r, q, T, sk, iv); });
    m.def("loss_heston", [](double k, double t, double sv, double rh, double v0, double S0, double r, double q, double T, std::vector<double> sk, std::vector<double> iv) { double pen = (2.0*k*t<=sv*sv)?1e7:0.0; return loss_normalized_robust([&](Complex u){return cf_heston(u,r,q,T,k,t,sv,rh,v0);}, S0, r, q, T, sk, iv)+pen; });
    m.def("fft_pricer_bs", [](double S, double r, double q, double si, double T, double a, int n_ov, double e_ov, std::string ty) { double e = (T<0.1)?0.5:0.25; int N = (T<0.1)?8192:4096; if (n_ov>0) N=n_ov; if (e_ov>0) e=e_ov; return pricing_core_extended([&](Complex u){return cf_bs(u,r,q,si,T);}, S, r, q, T, a, N, e, ty); });
    m.def("fft_pricer_heston", [](double S, double r, double q, double T, double k, double t, double sv, double rh, double v0, double a, int n_ov, double e_ov, std::string ty) { double e = (T<0.1)?0.5:0.25; int N = (T<0.1)?8192:4096; if (n_ov>0) N=n_ov; if (e_ov>0) e=e_ov; return pricing_core_extended([&](Complex u){return cf_heston(u,r,q,T,k,t,sv,rh,v0);}, S, r, q, T, a, N, e, ty); });
    m.def("fft_pricer_merton", [](double S, double r, double q, double si, double T, double l, double mj, double sj, double a, int n_ov, double e_ov, std::string ty) { double e = (T<0.1)?0.5:0.25; int N = (T<0.1)?8192:4096; if (n_ov>0) N=n_ov; if (e_ov>0) e=e_ov; return pricing_core_extended([&](Complex u){return cf_merton(u,r,q,si,T,l,mj,sj);}, S, r, q, T, a, N, e, ty); });
    m.def("fft_pricer_vg", [](double S, double r, double q, double T, double si, double n, double tv, double a, int n_ov, double e_ov, std::string ty) { double e = (T<0.1)?0.5:0.25; int N = (T<0.1)?8192:4096; if (n_ov>0) N=n_ov; if (e_ov>0) e=e_ov; return pricing_core_extended([&](Complex u){return cf_vg(u,r,q,T,si,n,tv);}, S, r, q, T, a, N, e, ty); });
}
