// reduction.cxx

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
#include <cctype>

using namespace Eigen;
using namespace std;

MatrixXd read_ecsv(ifstream& file, size_t cols) {
    string line;
    vector<vector<double>> rows;

    while (getline(file, line)) {
        if (line.empty()) continue;
        if (!isdigit((unsigned char)line[0]) && line[0] != '-' && line[0] != '+') continue;

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

void rotate_columns(MatrixXd& M, int p, int q, double c, double s) {
    VectorXd Mp = M.col(p);
    VectorXd Mq = M.col(q);

    M.col(p) = c * Mp + s * Mq;
    M.col(q) = -s * Mp + c * Mq;
}

void rotate_rows(MatrixXd& M, int p, int q, double c, double s) {
    RowVectorXd Mp = M.row(p);
    RowVectorXd Mq = M.row(q);

    M.row(p) = c * Mp + s * Mq;
    M.row(q) = -s * Mp + c * Mq;
}

double max_offdiag(MatrixXd const& M) {
    double max_value = 0.0;

    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            if (i != j) {
                max_value = max(max_value, abs(M(i, j)));
            }
        }
    }

    return max_value;
}

void apply_left_2x2(MatrixXd& S, MatrixXd& Uj, int p, int q, Matrix2d const& L) {
    RowVectorXd Sp = S.row(p);
    RowVectorXd Sq = S.row(q);

    S.row(p) = L(0, 0) * Sp + L(1, 0) * Sq;
    S.row(q) = L(0, 1) * Sp + L(1, 1) * Sq;

    VectorXd Up = Uj.col(p);
    VectorXd Uq = Uj.col(q);

    Uj.col(p) = L(0, 0) * Up + L(1, 0) * Uq;
    Uj.col(q) = L(0, 1) * Up + L(1, 1) * Uq;
}

size_t svd_jacobi(
    MatrixXd const& A, MatrixXd& U, VectorXd& s, MatrixXd& V,
    double max_off_diagonal, size_t max_sweeps = 100
) {
    const int m = A.rows();
    const int n = A.cols();

    MatrixXd Rfull = A;
    MatrixXd Q = MatrixXd::Identity(m, m);

    for (int j = 0; j < n; ++j) {
        for (int i = m - 1; i > j; --i) {
            double x = Rfull(i - 1, j);
            double y = Rfull(i, j);

            if (abs(y) < 1e-14) continue;

            double r = hypot(x, y);
            double c = x / r;
            double sn = y / r;

            rotate_rows(Rfull, i - 1, i, c, sn);
            rotate_columns(Q, i - 1, i, c, sn);
        }
    }

    MatrixXd S = Rfull.topRows(n);
    MatrixXd Uj = MatrixXd::Identity(n, n);
    V = MatrixXd::Identity(n, n);

    size_t sweep = 0;

    for (; sweep < max_sweeps; ++sweep) {
        double off = max_offdiag(S);

        if (off < max_off_diagonal) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double a = S(p, p);
                double b = S(p, q);
                double c = S(q, p);
                double d = S(q, q);

                if (max(abs(b), abs(c)) < max_off_diagonal) continue;

                Matrix2d C;
                C << a, b,
                     c, d;

                double m11 = a * a + c * c;
                double m22 = b * b + d * d;
                double m12 = a * b + c * d;

                double phi = 0.5 * atan2(2.0 * m12, m11 - m22);
                double cg = cos(phi);
                double sg = sin(phi);

                Matrix2d G;
                G << cg, -sg,
                     sg,  cg;

                Matrix2d T = C * G;

                double sigma1 = T.col(0).norm();
                double sigma2 = T.col(1).norm();

                if (sigma1 < 1e-14 || sigma2 < 1e-14) continue;

                Matrix2d L;
                L.col(0) = T.col(0) / sigma1;
                L.col(1) = T.col(1) / sigma2;

                rotate_columns(S, p, q, cg, sg);
                rotate_columns(V, p, q, cg, sg);

                apply_left_2x2(S, Uj, p, q, L);
            }
        }
    }

    MatrixXd Qthin = Q.leftCols(n);
    U = Qthin * Uj;

    s.resize(n);

    for (int i = 0; i < n; ++i) {
        s(i) = S(i, i);

        if (s(i) < 0.0) {
            s(i) = -s(i);
            U.col(i) *= -1.0;
        }
    }

    vector<int> idx(n);

    for (int i = 0; i < n; ++i) {
        idx[i] = i;
    }

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
    A = np.clip(A, 0.0, 1.0)
    ax.imshow(A, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

plt.savefig("reduction.pdf", bbox_inches="tight")
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

        MatrixXd A5 = rank_k_approx(U, s, V, 5);
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
        }
        else {
            cout << "Run manually: python3 make_reduction_pdf.py" << endl;
        }

    }
    catch (exception const& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}