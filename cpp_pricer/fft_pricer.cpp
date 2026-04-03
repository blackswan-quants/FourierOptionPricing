#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>
#include <vector>
#include <cmath>

namespace py = pybind11;

using Complex = std::complex<double>;
const double PI = 3.14159265358979323846;

// Basic Radix-2 inplace Cooley-Tukey FFT
void fft(std::vector<Complex>& x) {
    size_t N = x.size();
    if (N <= 1) return;

    // Bit-reverse permutation
    size_t j = 0;
    for (size_t i = 1; i < N; ++i) {
        size_t bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }

    // Cooley-Tukey
    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = -2 * PI / len;
        Complex wlen(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < N; i += len) {
            Complex w(1.0, 0.0);
            for (size_t k = 0; k < len / 2; ++k) {
                Complex u = x[i + k];
                Complex v = x[i + k + len / 2] * w;
                x[i + k] = u + v;
                x[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// Characteristic function for Black-Scholes
Complex cf_bs(Complex u, double S0, double r, double sigma, double T) {
    double mu = std::log(S0) + (r - 0.5 * sigma * sigma) * T;
    return std::exp(Complex(0.0, 1.0) * u * mu - 0.5 * sigma * sigma * u * u * T);
}

// Characteristic function for Heston
Complex cf_heston(Complex u, double S0, double r, double T, 
                  double kappa, double theta, double sigma_v, double rho, double v0) {
    Complex i(0.0, 1.0);
    double a = kappa * theta;
    
    Complex d = std::sqrt(std::pow(rho * sigma_v * i * u - kappa, 2) + sigma_v * sigma_v * (i * u + u * u));
    Complex b = kappa - rho * sigma_v * i * u;
    Complex g = (b - d) / (b + d);
    
    Complex exp_neg_dT = std::exp(-d * T);
    
    Complex C = (i * u * (std::log(S0) + r * T) 
                 + a / (sigma_v * sigma_v) * ((b - d) * T - 2.0 * std::log((1.0 - g * exp_neg_dT) / (1.0 - g))));
                 
    Complex D = (b - d) / (sigma_v * sigma_v) * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT));
    
    return std::exp(C + D * v0);
}

// FFT Engine for Black-Scholes
py::tuple fft_pricer_bs(double S0, double r, double sigma, double T,
                        double alpha, int N, double eta, std::string option_type) {
    
    std::vector<Complex> fft_input(N, 0.0);
    double lambd = 2 * PI / (N * eta);
    double b = 0.5 * N * lambd;
    
    double discount = std::exp(-r * T);
    
    for (int j = 0; j < N; ++j) {
        double v = j * eta;
        Complex u = v - Complex(0.0, alpha + 1.0);
        
        Complex phi_vals = cf_bs(u, S0, r, sigma, T);
        Complex denom = alpha * alpha + alpha - v * v + Complex(0.0, 2 * alpha + 1) * v;
        Complex psi = discount * phi_vals / denom;
        
        double w = (eta / 3.0) * (3.0 + (j % 2 == 0 ? -1.0 : 1.0));
        if (j == 0) w = eta / 3.0;
        
        fft_input[j] = std::exp(Complex(0.0, b * v)) * psi * w;
    }
    
    fft(fft_input);
    
    py::array_t<double> K_arr(N);
    py::array_t<double> values_arr(N);
    auto ptr_K = K_arr.mutable_unchecked<1>();
    auto ptr_values = values_arr.mutable_unchecked<1>();
    
    for (int m = 0; m < N; ++m) {
        double k = -b + m * lambd;
        double K_val = std::exp(k);
        ptr_K(m) = K_val;
        
        double call_val = std::exp(-alpha * k) * fft_input[m].real() / PI;
        
        if (option_type == "put" || option_type == "Put") {
            ptr_values(m) = call_val - S0 + K_val * discount;
        } else {
            ptr_values(m) = call_val;
        }
    }
    
    return py::make_tuple(K_arr, values_arr);
}

// FFT Engine for Heston
py::tuple fft_pricer_heston(double S0, double r, double T, 
                            double kappa, double theta, double sigma_v, double rho, double v0,
                            double alpha, int N, double eta, std::string option_type) {
    
    std::vector<Complex> fft_input(N, 0.0);
    double lambd = 2 * PI / (N * eta);
    double b = 0.5 * N * lambd;
    
    double discount = std::exp(-r * T);
    
    for (int j = 0; j < N; ++j) {
        double v = j * eta;
        Complex u = v - Complex(0.0, alpha + 1.0);
        
        Complex phi_vals = cf_heston(u, S0, r, T, kappa, theta, sigma_v, rho, v0);
        Complex denom = alpha * alpha + alpha - v * v + Complex(0.0, 2 * alpha + 1) * v;
        Complex psi = discount * phi_vals / denom;
        
        double w = (eta / 3.0) * (3.0 + (j % 2 == 0 ? -1.0 : 1.0));
        if (j == 0) w = eta / 3.0;
        
        fft_input[j] = std::exp(Complex(0.0, b * v)) * psi * w;
    }
    
    fft(fft_input);
    
    py::array_t<double> K_arr(N);
    py::array_t<double> values_arr(N);
    auto ptr_K = K_arr.mutable_unchecked<1>();
    auto ptr_values = values_arr.mutable_unchecked<1>();
    
    for (int m = 0; m < N; ++m) {
        double k = -b + m * lambd;
        double K_val = std::exp(k);
        ptr_K(m) = K_val;
        
        double call_val = std::exp(-alpha * k) * fft_input[m].real() / PI;
        
        if (option_type == "put" || option_type == "Put") {
            ptr_values(m) = call_val - S0 + K_val * discount;
        } else {
            ptr_values(m) = call_val;
        }
    }
    
    return py::make_tuple(K_arr, values_arr);
}

PYBIND11_MODULE(cpp_pricer, m) {
    m.doc() = "C++ FFT Option Pricer module via Cooley-Tukey Radix-2";
    
    m.def("fft_pricer_bs", &fft_pricer_bs, "Carr-Madan FFT Pricer for Black-Scholes",
          py::arg("S0"), py::arg("r"), py::arg("sigma"), py::arg("T"),
          py::arg("alpha") = 1.5, py::arg("N") = 4096, py::arg("eta") = 0.25, 
          py::arg("option_type") = "call");
          
    m.def("fft_pricer_heston", &fft_pricer_heston, "Carr-Madan FFT Pricer for Heston",
          py::arg("S0"), py::arg("r"), py::arg("T"), 
          py::arg("kappa"), py::arg("theta"), py::arg("sigma_v"), py::arg("rho"), py::arg("v0"),
          py::arg("alpha") = 1.5, py::arg("N") = 4096, py::arg("eta") = 0.25, 
          py::arg("option_type") = "call");
}
