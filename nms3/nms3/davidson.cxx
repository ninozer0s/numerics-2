#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
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
    // anharmonic oscillator so potential has higher term than square
    // -> potential is not a simple parabola anymore and the oscillation frequency depends on the amplitude
    // also energy levels not equally spaced
    MatrixXd H = MatrixXd::Zero((int)n, (int)n);

    // finite differences (zentrales differenzenverfahren) approximating derivative (from left and right both)
    double kinetic_factor = -1.0 / (2.0 * dx * dx);

    for (size_t i = 0; i < n; ++i) {
        H((int)i, (int)i) += kinetic_factor * (-2.0);

        // periodische randbedingungen!
        H((int)i, (int)((i + 1) % n)) += kinetic_factor;
        H((int)i, (int)((i + n - 1) % n)) += kinetic_factor;
    }

    for (size_t i = 0; i < n; ++i) {
        // potential term, not approximating derivative, therefore only on main diagonal
        double grid_index = static_cast<double>(i) - static_cast<double>(n) / 2.0;
        H((int)i, (int)i) += pow(dx, 4) * pow(grid_index, 4) / 24.0;
    }

    return H;
}

// Davidson algorithm great for finding lowest eigenvalues and eigenvectors of sparse 
DavidsonResult davidson(const MatrixXd& H,
    const VectorXd& v1,
    size_t K,
    size_t L,
    size_t iterations) {

    const size_t n = (size_t)H.rows();
    
    // P is the diagonal entry of H
    // to approximate H in the inverse
    VectorXd P = H.diagonal();

    // we use a first starting vector and iteratively improve the directions
    // rayleigh coefficient here, estimate of our first eigenvalue
    // numerator = expected value of the energy
    double lambda1 = (v1.transpose() * H * v1)(0, 0) /
        (v1.transpose() * v1)(0, 0);

    // make our matrix V, updating it with the eigenvectors iteratively
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
            // the residual which we get from an eigenvalue problem equation
            // if our vector is the exact eigenvector of H, r becomes 0
            VectorXd r = H * V.col((int)k) - lambdas((int)k) * V.col((int)k);
            VectorXd delta_v((int)n);

            for (size_t i = 0; i < n; ++i) {

                // bauen den precoditioner? (-P - lambda I)^-1 auf
                // um die Korrektur des Vektors v zu berechnen
                // P kommt von der Diagonale von der Matrix H
                // direktes Inverses von H ist zu leistungsinstensiv ofc
                // -> P - lambda*I ist eine Diagonalmatrix
                // 
                double denominator = P((int)i) - lambdas((int)k);

                if (fabs(denominator) < 1e-12) { // important because in the davidson method, the denominator can go against 0
                    // if the ritz value is equal a diagonal element
                    delta_v((int)i) = 0.0;
                }
                else {
                    // mathematische Korrektur: delta_v ~ -(P - lambda I)^-1 * r
                    delta_v((int)i) = -r((int)i) / denominator;
                }
            }
            // adding new directions
            W.col((int)(M + k)) = delta_v;
        }

        // our expanded space W with all the eigenvectors and corrections
        // orthogonalise it (QR decomposition) and get the basis U 
        // because we only need the orthonormal base of the space that the matrix W makes
        HouseholderQR<MatrixXd> W_QR(W);

        MatrixXd U =
            W_QR.householderQ() *
            MatrixXd::Identity((int)n, (int)(M + Kprime));

        // Ritz values and vectors
        // project the matrix H into this space built of our eigenvectors
        // reduce the "big problem" onto a "small problem"
        MatrixXd J = U.transpose() * H * U;

        // J is as big as our amount of test eigenvectors, with L = 40
        // get the best approximation in the current space
        SelfAdjointEigenSolver<MatrixXd> J_eigensystem(J);

        size_t newM = min(M + Kprime, L);

        // the eigenvalues of J, the Ritz values, are the currect approximations of the eigenvalues of H
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
        }
        else {
            out << " nan";
        }

        if (current.size() > 1) {
            double e1 = fabs(current.coeff(1) - exact_values.coeff(1));
            out << " " << e1;
        }
        else {
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

names = psi0.dtype.names[2:]
num_plots = len(names)
for i, name in enumerate(names):
    progress = (i / (num_plots - 1) if num_plots > 1 else 0)
    blue_val = 1.0 - (0.5 * progress)  
    other_val = 0.7 * (1 - progress)
    ax[0].plot(psi0["x"], psi0[name], linewidth=0.8, label=name, color=(other_val, other_val, blue_val))

ax[0].plot(psi0["x"], psi0["exact"], color="red", linestyle="--", linewidth=1.4, label="exact")
ax[0].set_title("ground state psi0")
ax[0].set_xlabel("x")
ax[0].set_ylabel("psi0")
ax[0].legend(fontsize=7, ncol=4)
ax[0].set_xlim(-5,5)



names1 = psi1.dtype.names[2:]
num_plots1 = len(names1)
for i, name in enumerate(names1):
    progress = (i / (num_plots1 - 1) if num_plots1 > 1 else 0)
    blue_val = 1.0 - (0.5 * progress) 
    other_val = 0.7 * (1 - progress)
    ax[1].plot(psi1["x"], psi1[name], linewidth=0.8, label=name, color=(other_val, other_val, blue_val))

ax[1].plot(psi1["x"], psi1["exact"], color="red", linestyle="--", linewidth=1.4, label="exact")
ax[1].set_title("first excited state psi1")
ax[1].set_xlabel("x")
ax[1].set_ylabel("psi1")
ax[1].set_xlim(-5,5)
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
    // bestimmen der Gittergröße und Gitterweite
    const size_t n = 256;
    // ist der Abstand zwischen diskretisierten Punkten der Ableitungen
    // -> finite differences
    const double dx = 0.15;

    const size_t K = 2;
    // räumliche Ausdehnung des Systems
    const size_t L = 40;
    const size_t iterations = 60;

    MatrixXd H = build_hamiltonian(n, dx);

    VectorXd x((int)n);

    for (size_t i = 0; i < n; ++i) {
        x((int)i) =
            (static_cast<double>(i) - static_cast<double>(n) / 2.0) * dx;
    }

    VectorXd v1 = VectorXd::Zero((int)n);

    // Random start vector, aber nur im physikalisch relevanten Bereich.
    // Das ist kein Gaußvektor. Die Werte sind zufällig.
    mt19937 rng(12345);
    uniform_real_distribution<double> dist(-1.0, 1.0);

    const double cutoff = 4.5;

    for (size_t i = 0; i < n; ++i) {
        if (fabs(x((int)i)) <= cutoff) {
            v1((int)i) = dist(rng);
        }
        else {
            v1((int)i) = 0.0;
        }
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