#include <cmath>
#include <iostream>

double poisson_pmf(int k, double lambda) {
    double e = std::exp(1);
    double numerator = std::pow(lambda, k) * std::pow(e, -lambda);
    double denominator = 1;
    for(int i = 1; i <= k; ++i) {
        denominator *= i;
    }
    return numerator / denominator;
}

float gamma_pdf(int x, float alpha = 1.0, float beta = 16.0) {
    float e = std::exp(1);
    float numerator = std::pow(x, alpha - 1) * std::pow(e, -x / beta);
    float denominator = std::pow(beta, alpha) * std::tgamma(alpha);
    return numerator / denominator;
}

int main() {
    float lambda = 1.0, alpha = 1.0, beta = 16;

    for (int i = 0; i < 1; ++i) {
        float sum = 0;
        for(int k = i; k < 28 + i; ++k) {
            sum += gamma_pdf(k, alpha, beta);
        }
        std::cout << "Sum of P(X = k) for i = 0 to 10: " << sum << std::endl;
    }

    return 0;
}
