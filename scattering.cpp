#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace Eigen;
using namespace std;

// ------------------------------------------------------------
// scattering matrix S
// S = (1-kappa)/2 * tridiagonal matrix with 1 on off-diagonals
// ------------------------------------------------------------
MatrixXd make_scattering_matrix(int n, double kappa) {
    if (n <= 0) {
        throw invalid_argument("n must be positive");
    }
    if (kappa < 0.0 || kappa > 1.0) {
        throw invalid_argument("kappa must be in [0,1]");
    }

    MatrixXd S = MatrixXd::Zero(n, n);
    double a = (1.0 - kappa) / 2.0;

    for (int j = 0; j < n; ++j) {
        if (j > 0)     S(j - 1, j) = a; // photons scatter left
        if (j < n - 1) S(j + 1, j) = a; // photons scatter right
    }

    return S;
}

// ------------------------------------------------------------
// compute steady state: (I - S)^(-1) * i
// ------------------------------------------------------------
VectorXd steady_state(MatrixXd const& S, VectorXd const& i) {
    if (S.rows() != S.cols()) {
        throw invalid_argument("S must be square");
    }
    if (S.rows() != i.size()) {
        throw invalid_argument("dimension mismatch between S and i");
    }

    MatrixXd I = MatrixXd::Identity(S.rows(), S.cols());
    return (I - S).inverse() * i;
}

int main() {
    // reasonable example values
    int n = 100;
    double kappa = 0.2;

    MatrixXd S = make_scattering_matrix(n, kappa);

    // incoming current from the left end
    VectorXd i = VectorXd::Zero(n);
    i(0) = 1.0;

    VectorXd p = steady_state(S, i);

    // print first few values
    cout << "steady state:\n";
    for (int j = 0; j < min(n, 10); ++j) {
        cout << "p(" << j << ") = " << p(j) << '\n';
    }

    // save data for plotting
    ofstream data("steady_state.dat");
    data << setprecision(16);
    for (int j = 0; j < n; ++j) {
        data << j << " " << p(j) << "\n";
    }
    data.close();

    // optional: create gnuplot script for PDF output
    ofstream gp("plot_steady_state.gnuplot");
    gp << "set terminal pdfcairo\n";
    gp << "set output 'steady_state.pdf'\n";
    gp << "set xlabel 'site index'\n";
    gp << "set ylabel 'steady state photons'\n";
    gp << "set title 'Steady state of scattering model'\n";
    gp << "plot 'steady_state.dat' using 1:2 with lines lw 2 title 'n="
       << n << ", kappa=" << kappa << "'\n";
    gp.close();

    cout << "\nData written to steady_state.dat\n";
    cout << "Gnuplot script written to plot_steady_state.gnuplot\n";
    cout << "Run: gnuplot plot_steady_state.gnuplot\n";

    return 0;
}