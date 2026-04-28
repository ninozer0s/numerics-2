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
    // check out, good additional sources as explanations (helped us) 
    // https://www.cs.utexas.edu/~inderjit/public_papers/HLA_SVD.pdf 
    // https://xilinx.github.io/Vitis_Libraries/quantitative_finance/2019.2/guide_L1/SVD/SVD.html
    // As in the source: "Jacobi methods apply plane rotations to the entire matrix A ...
    // Two-sided Jacboi methods iteratively apply rotations on both sides of matrix A to bring it to diagonal form."
    
    const int m = A.rows();
    const int n = min(A.rows(), A.cols());

    MatrixXd S = A;
    MatrixXd P = MatrixXd::Identity(A.rows(), A.rows());
    MatrixXd Q = MatrixXd::Identity(A.cols(), A.cols());

    size_t sweep = 0;
    for (; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;

        for (int k = 0; k < n - 1; ++k) {
            for (int l = k + 1; l < n; ++l) {
                
                // we want to diagonalize, iteratively get the elements for angle calculation
                double Skk = S(k, k);
                double Sll = S(l, l);
                double Skl = S(k, l);
                double Slk = S(l, k);

                max_off = max({ max_off, abs(Skl), abs(Slk) });
                if (max(abs(Skl), abs(Slk)) < max_off_diagonal) continue;

                // alpha, the first in the line of rotation angles
                // left rotation, helps symmetrize the elements
                // tan(alpha) = (S_l^k - S_k^l) / (S_k^k + S_l^l) as in the cheat sheet
                double alpha = atan2(Slk - Skl, Skk + Sll);
                double ca = cos(alpha);
                double sa = sin(alpha);

                // T = G(k, l, alpha) * S    next step matrix
                // the formulae and the process of each matrix with cosine and sine we use from the cheat sheet
                // we know this constellation from the givens rotation
                VectorXd row_k = S.row(k);
                VectorXd row_l = S.row(l);
                S.row(k) = ca * row_k - sa * row_l;
                S.row(l) = sa * row_k + ca * row_l;

                // beta, the second rotation angle
                // right rotation, diagonalize 2x2 block
                // tan(2*beta) = 2*T_k^l / (T_l^l - T_k^k)
                double Tkk = S(k, k);
                double Tll = S(l, l);
                double Tkl = S(k, l);
                double beta = 0.5 * atan2(2.0 * Tkl, Tll - Tkk);
                double cb = cos(beta);
                double sb = sin(beta);

                // S_new = G(beta) * T * G*(beta) 
                // apply left G(beta)
                row_k = S.row(k);
                row_l = S.row(l);
                S.row(k) = cb * row_k - sb * row_l;
                S.row(l) = sb * row_k + cb * row_l;

                // apply right G*(beta)
                VectorXd col_k = S.col(k);
                VectorXd col_l = S.col(l);
                S.col(k) = cb * col_k - sb * col_l;
                S.col(l) = sb * col_k + cb * col_l;

                // P and Q add up the rotations -> so they become U and V
                // P = P * G*(alpha + beta)
                double cab = cos(alpha + beta);
                double sab = sin(alpha + beta);
                for (int i = 0; i < A.rows(); ++i) {
                    double Pik = cab * P(i, k) - sab * P(i, l);
                    double Pil = sab * P(i, k) + cab * P(i, l);
                    P(i, k) = Pik;
                    P(i, l) = Pil;
                }
                VectorXd Pk = P.col(k);
                VectorXd Pl = P.col(l);
                P.col(k) = cab * Pk - sab * Pl;
                P.col(l) = sab * Pk + cab * Pl;

                // Q = Q * G*(beta
                VectorXd Qk = Q.col(k);
                VectorXd Ql = Q.col(l);
                Q.col(k) = cb * Qk - sb * Ql;
                Q.col(l) = sb * Qk + cb * Ql;
            }
        }
        if (max_off < max_off_diagonal) break;
    }

    // make sure al singular values are positive
    // for that we'd have to also compensate with that column in the matrix by -1 multiplication
    s.resize(n);
    U.resize(A.rows(), n);
    V.resize(A.cols(), n);

    for (int k = 0; k < n; ++k) {
        double val = S(k, k);
        s(k) = abs(val);
        double sign = (val >= 0) ? 1.0 : -1.0;
        U.col(k) = P.col(k) * sign;
        V.col(k) = Q.col(k);
    }

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