#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace Eigen;
using namespace std;

MatrixXd make_scattering_matrix(int n, double kappa) {
    MatrixXd S = MatrixXd::Zero(n, n);
    double a = (1.0 - kappa) / 2.0;

    for (int j = 0; j < n; ++j) {
        if (j > 0)     S(j - 1, j) = a; // photons scatter left
        if (j < n - 1) S(j + 1, j) = a; // photons scatter right
    }

    return S;
}


VectorXd steady_state(MatrixXd const& S, VectorXd const& i) {
    MatrixXd I = MatrixXd::Identity(S.rows(), S.cols());
    return (I - S).inverse() * i;
}

VectorXd solve_minres(MatrixXd const& A, VectorXd const& b, size_t r) {
    int n = A.rows();    
    MatrixXd Q = MatrixXd::Zero(n, r);

    VectorXd v = b;
    for (size_t j = 0; j < r; ++j) {
        Q.col(j) = v;
        v = A * v; 
    }

    HouseholderQR<MatrixXd> QR(Q);
    MatrixXd Q_ortho = QR.householderQ() * MatrixXd::Identity(n, r);

    MatrixXd AQ = A * Q_ortho;
    VectorXd y = AQ.householderQr().solve(b); 

    return Q_ortho * y;
}

int main() { 
    // reasonable example values
    int n = 100;
    double kappa = 0.1;
    size_t r = 20;

    MatrixXd S = make_scattering_matrix(n, kappa);
    MatrixXd Other_matrix = MatrixXd::Identity(n, n) - S;

    // incoming current from the left end
    VectorXd i = VectorXd::Zero(n);
    i(0) = 1.0;

    VectorXd p = steady_state(S, i);
    VectorXd p_min = solve_minres(Other_matrix, i, r);

    // print first few values instead of debugging
    cout << "steady state:\n";
    for (int j = 0; j < min(n, 10); ++j) {
        cout << "p(" << j << ") = " << p(j) << " " << p_min(j) << '\n';
    }

    // save data for plotting
    ofstream data("steady_state.dat");
    data << setprecision(16);
    for (int j = 0; j < n; ++j) {
        data << j << " " << p(j) << " " << p_min(j) << "\n";
    }
    data.close();

    // a good to attempt at creating gnuplot
    ofstream gp("plot_steady_state.gnuplot");
    gp << "set terminal pdfcairo\n";
    gp << "set output 'steady_state.pdf'\n";
    gp << "set xlabel 'site index'\n";
    gp << "set ylabel 'steady state photons'\n";
    gp << "set title 'Steady state of scattering model'\n";
    gp << "plot 'steady_state.dat' using 1:2 with lines lw 2 title 'n="
       << n << ", kappa=" << kappa << "'\n";
    gp << "     'steady_state.dat' u 1:3 w lines dt 2 title 'MINRES (r=" << r << ")'\n";
    gp.close();

    cout << "\nData written to steady_state.dat\n";
    cout << "Gnuplot script written to plot_steady_state.gnuplot\n";
    cout << "Run: gnuplot plot_steady_state.gnuplot\n";

    return 0;
}