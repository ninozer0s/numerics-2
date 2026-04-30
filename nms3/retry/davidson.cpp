#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

struct DavidsonResult {
    VectorXd eigenvalues;
    MatrixXd eigenvectors;
    vector<VectorXd> lambda_history;
    vector<MatrixXd> vector_history;
};

MatrixXd build_hamiltonian(size_t n, double dx) {
    MatrixXd H = MatrixXd::Zero((int)n, (int)n);

    double kinetic_factor = -1.0 / (2.0 * dx * dx);

    for (size_t i = 0; i < n; ++i) {
        H((int)i, (int)i) += kinetic_factor * (-2.0);

        // Periodische Randbedingungen, wie in der Matrix der Angabe:
        H((int)i, (int)((i + 1) % n)) += kinetic_factor;
        H((int)i, (int)((i + n - 1) % n)) += kinetic_factor;
    }

    for (size_t i = 0; i < n; ++i) {
        double grid_index = static_cast<double>(i) - static_cast<double>(n) / 2.0;
        H((int)i, (int)i) += pow(dx, 4) * pow(grid_index, 4) / 24.0;
    }

    return H;
}

DavidsonResult davidson(const MatrixXd& H,
                        const VectorXd& v1,
                        size_t K,
                        size_t L,
                        size_t iterations) {
    const size_t n = (size_t)H.rows();

    VectorXd P = H.diagonal();

    double lambda1 = (v1.transpose() * H * v1)(0, 0) /
                     (v1.transpose() * v1)(0, 0);

    MatrixXd V((int)n, 1);
    V.col(0) = v1.normalized();

    VectorXd lambdas(1);
    lambdas(0) = lambda1;

    vector<VectorXd> lambda_history;
    vector<MatrixXd> vector_history;

    lambda_history.push_back(lambdas);
    vector_history.push_back(V);

    size_t iteration = 0;

    while (iteration < iterations) {
        size_t M = (size_t)V.cols();
        size_t Kprime = min(M, K);

        MatrixXd W((int)n, (int)(M + Kprime));

        for (size_t k = 0; k < M; ++k) {
            W.col((int)k) = V.col((int)k);
        }

        for (size_t k = 0; k < Kprime; ++k) {
            VectorXd r = H * V.col((int)k) - lambdas((int)k) * V.col((int)k);
            VectorXd delta_v((int)n);

            for (size_t i = 0; i < n; ++i) {
                double denominator = P((int)i) - lambdas((int)k);

                if (fabs(denominator) < 1e-12) {
                    delta_v((int)i) = 0.0;
                } else {
                    delta_v((int)i) = -r((int)i) / denominator;
                }
            }

            W.col((int)(M + k)) = delta_v;
        }

        HouseholderQR<MatrixXd> W_QR(W);

        MatrixXd U =
            W_QR.householderQ() *
            MatrixXd::Identity((int)n, (int)(M + Kprime));

        MatrixXd J = U.transpose() * H * U;

        SelfAdjointEigenSolver<MatrixXd> J_eigensystem(J);

        size_t newM = min(M + Kprime, L);

        lambdas = J_eigensystem.eigenvalues().head((int)newM);
        V = U * J_eigensystem.eigenvectors().leftCols((int)newM);

        lambda_history.push_back(lambdas);
        vector_history.push_back(V);

        iteration += 1;
    }

    size_t output_size = min((size_t)V.cols(), K);

    DavidsonResult result;
    result.eigenvalues = lambdas.head((int)output_size);
    result.eigenvectors = V.leftCols((int)output_size);
    result.lambda_history = lambda_history;
    result.vector_history = vector_history;

    return result;
}

void write_vector_data(const string& filename,
                       const VectorXd& x,
                       const vector<MatrixXd>& vector_history,
                       const MatrixXd& exact_vectors,
                       size_t state) {
    ofstream out(filename);
    out << setprecision(16);

    out << "x exact";

    for (size_t r = 0; r < vector_history.size(); ++r) {
        if (state < (size_t)vector_history.at(r).cols()) {
            out << " r" << r;
        }
    }

    out << "\n";

    for (int i = 0; i < x.size(); ++i) {
        out << x(i) << " " << exact_vectors(i, (int)state);

        for (size_t r = 0; r < vector_history.size(); ++r) {
            const MatrixXd& Vr = vector_history.at(r);

            if (state < (size_t)Vr.cols()) {
                VectorXd v = Vr.col((int)state);

                if (v.dot(exact_vectors.col((int)state)) < 0.0) {
                    v = -v;
                }

                out << " " << v(i);
            }
        }

        out << "\n";
    }
}

void write_error_data(const string& filename,
                      const vector<VectorXd>& lambda_history,
                      const VectorXd& exact_values) {
    ofstream out(filename);
    out << setprecision(16);

    out << "r E0_error E1_error\n";

    for (size_t row = 0; row < lambda_history.size(); ++row) {
        const VectorXd& current = lambda_history.at(row);

        out << row;

        if (current.size() > 0) {
            double e0 = fabs(current.coeff(0) - exact_values.coeff(0));
            out << " " << e0;
        } else {
            out << " nan";
        }

        if (current.size() > 1) {
            double e1 = fabs(current.coeff(1) - exact_values.coeff(1));
            out << " " << e1;
        } else {
            out << " nan";
        }

        out << "\n";
    }
}

void write_plot_script() {
    ofstream py("make_davidson_pdf.py");

    py << R"PY(
import numpy as np
import matplotlib.pyplot as plt

psi0 = np.genfromtxt("psi0.dat", names=True)
psi1 = np.genfromtxt("psi1.dat", names=True)
err = np.genfromtxt("errors.dat", names=True)

fig, ax = plt.subplots(3, 1, figsize=(8, 10))

for name in psi0.dtype.names[2:]:
    ax[0].plot(psi0["x"], psi0[name], linewidth=0.8, label=name)

ax[0].plot(psi0["x"], psi0["exact"], linestyle="--", linewidth=1.4, label="exact")
ax[0].set_title("ground state psi0")
ax[0].set_xlabel("x")
ax[0].set_ylabel("psi0")
ax[0].legend(fontsize=7, ncol=4)

for name in psi1.dtype.names[2:]:
    ax[1].plot(psi1["x"], psi1[name], linewidth=0.8, label=name)

ax[1].plot(psi1["x"], psi1["exact"], linestyle="--", linewidth=1.4, label="exact")
ax[1].set_title("first excited state psi1")
ax[1].set_xlabel("x")
ax[1].set_ylabel("psi1")
ax[1].legend(fontsize=7, ncol=4)

ax[2].semilogy(err["r"], err["E0_error"], marker="o", label="|E0 - exact|")
ax[2].semilogy(err["r"], err["E1_error"], marker="o", label="|E1 - exact|")
ax[2].set_title("eigenvalue errors")
ax[2].set_xlabel("Davidson iteration r")
ax[2].set_ylabel("absolute error")
ax[2].legend()

fig.tight_layout()
fig.savefig("davidson.pdf")
)PY";
}

int main() {
    const size_t n = 256;
    const double dx = 0.15;

    const size_t K = 2;
    const size_t L = 30;
    const size_t iterations = 30;

    MatrixXd H = build_hamiltonian(n, dx);

    VectorXd x((int)n);

    for (size_t i = 0; i < n; ++i) {
        x((int)i) =
            (static_cast<double>(i) - static_cast<double>(n) / 2.0) * dx;
    }

    VectorXd v1((int)n);

    for (size_t i = 0; i < n; ++i) {
        double xi = x((int)i);
        v1((int)i) = exp(-0.5 * xi * xi) * (1.0 + 0.2 * xi);
    }

    v1.normalize();

    DavidsonResult result = davidson(H, v1, K, L, iterations);

    SelfAdjointEigenSolver<MatrixXd> exact(H);

    write_vector_data("psi0.dat", x, result.vector_history, exact.eigenvectors(), 0);
    write_vector_data("psi1.dat", x, result.vector_history, exact.eigenvectors(), 1);
    write_error_data("errors.dat", result.lambda_history, exact.eigenvalues());
    write_plot_script();

    cout << setprecision(16);

    cout << "Davidson eigenvalues:\n";
    for (int i = 0; i < result.eigenvalues.size(); ++i) {
        cout << result.eigenvalues(i) << "\n";
    }

    cout << "Exact eigenvalues:\n";
    for (size_t i = 0; i < K; ++i) {
        cout << exact.eigenvalues().coeff((int)i) << "\n";
    }

    system("python3 make_davidson_pdf.py");

    return 0;
}