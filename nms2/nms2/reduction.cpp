#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>

using namespace Eigen;
using namespace std;

MatrixXd read_ecsv(ifstream& file, size_t cols) {
    string line;
    vector<vector<double>> rows;

    while (getline(file, line)) {
        if (line.empty()) continue;
        if (!isdigit(line[0]) && line[0] != '-' && line[0] != '+') continue;

        replace(line.begin(), line.end(), ',', ' ');

        istringstream stream(line);
        vector<double> row;
        double x;

        while (stream >> x) row.push_back(x);

        if (row.size() != cols) {
            throw runtime_error("Wrong number of columns in input row.");
        }

        rows.push_back(row);
    }

    MatrixXd A(rows.size(), cols);

    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < cols; ++j)
            A(i, j) = rows[i][j];

    return A;
}

void write_matrix(string const& filename, MatrixXd const& A) {
    ofstream out(filename);
    out << setprecision(16);

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            out << A(i, j);
            if (j + 1 < A.cols()) out << " ";
        }
        out << "\n";
    }
}

size_t svd_jacobi(
    MatrixXd const& A, MatrixXd& U, VectorXd& s, MatrixXd& V,
    double max_off_diagonal, size_t max_sweeps = 100
) {
    const int m = A.rows();
    const int n = A.cols();

    MatrixXd B = A;
    V = MatrixXd::Identity(n, n);

    size_t sweep = 0;

    for (; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double app = B.col(p).squaredNorm();
                double aqq = B.col(q).squaredNorm();
                double apq = B.col(p).dot(B.col(q));

                max_off = max(max_off, abs(apq));

                if (abs(apq) < max_off_diagonal) continue;

                double tau = (aqq - app) / (2.0 * apq);
                double t = ((tau >= 0.0) ? 1.0 : -1.0)
                         / (abs(tau) + sqrt(1.0 + tau * tau));

                double c = 1.0 / sqrt(1.0 + t * t);
                double sn = c * t;

                VectorXd bp = B.col(p);
                VectorXd bq = B.col(q);

                B.col(p) = c * bp - sn * bq;
                B.col(q) = sn * bp + c * bq;

                VectorXd vp = V.col(p);
                VectorXd vq = V.col(q);

                V.col(p) = c * vp - sn * vq;
                V.col(q) = sn * vp + c * vq;
            }
        }

        if (max_off < max_off_diagonal) break;
    }

    s.resize(n);
    U = MatrixXd::Zero(m, n);

    for (int j = 0; j < n; ++j) {
        s(j) = B.col(j).norm();
        if (s(j) > 1e-14) {
            U.col(j) = B.col(j) / s(j);
        }
    }

    vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return s(a) > s(b);
    });

    VectorXd s_sorted(n);
    MatrixXd U_sorted(m, n);
    MatrixXd V_sorted(n, n);

    for (int i = 0; i < n; ++i) {
        s_sorted(i) = s(idx[i]);
        U_sorted.col(i) = U.col(idx[i]);
        V_sorted.col(i) = V.col(idx[i]);
    }

    s = s_sorted;
    U = U_sorted;
    V = V_sorted;

    return sweep + 1;
}

MatrixXd rank_k_approx(MatrixXd const& U, VectorXd const& s, MatrixXd const& V, int K) {
    return U.leftCols(K) * s.head(K).asDiagonal() * V.leftCols(K).transpose();
}

void write_python_plot_script() {
    ofstream py("make_reduction_pdf.py");

    py << R"(import numpy as np
import matplotlib.pyplot as plt

files = [
    ("Original", "original.dat"),
    ("K = 5", "reduction_K5.dat"),
    ("K = 10", "reduction_K10.dat"),
    ("K = 20", "reduction_K20.dat"),
]

fig, axes = plt.subplots(1, 4, figsize=(12, 4), constrained_layout=True)

for ax, (title, filename) in zip(axes, files):
    A = np.loadtxt(filename)
    ax.imshow(A, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


plt.savefig("reduction.pdf")
)";
}

int main() {
    try {
        ifstream file("bose.dat");

        if (!file) {
            throw runtime_error("Could not open bose.dat");
        }

        MatrixXd A = read_ecsv(file, 372);

        if (A.rows() != 497 || A.cols() != 372) {
            throw runtime_error("Expected matrix size 497 x 372.");
        }

        cout << "Read image matrix: " << A.rows() << " x " << A.cols() << endl;

        MatrixXd U, V;
        VectorXd s;

        size_t sweeps = svd_jacobi(A, U, s, V, 1e-8, 100);

        cout << "Jacobi sweeps: " << sweeps << endl;
        cout << "Largest singular values: "
             << s.head(10).transpose() << endl;

        MatrixXd A5  = rank_k_approx(U, s, V, 5);
        MatrixXd A10 = rank_k_approx(U, s, V, 10);
        MatrixXd A20 = rank_k_approx(U, s, V, 20);

        write_matrix("original.dat", A);
        write_matrix("reduction_K5.dat", A5);
        write_matrix("reduction_K10.dat", A10);
        write_matrix("reduction_K20.dat", A20);

        write_python_plot_script();

        int ret = system("python3 make_reduction_pdf.py");

        if (ret == 0) {
            cout << "Created reduction.pdf" << endl;
        } else {
            cout << "Run manually: python3 make_reduction_pdf.py" << endl;
        }

    } catch (exception const& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}