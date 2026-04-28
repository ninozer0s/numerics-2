#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace Eigen;
using namespace std;

MatrixXd make_scattering_matrix(int n, double kappa) {
    // MatrixXd --> X for dynamic runtime dimension as in the dimensions are set during running
    // d for double float
    MatrixXd S = MatrixXd::Zero(n, n);
    double a = (1.0 - kappa) / 2.0;

    for (int j = 0; j < n; ++j) {
        if (j > 0)     S(j - 1, j) = a; // photons scatter left
        if (j < n - 1) S(j + 1, j) = a; // photons scatter right
    } // goodnight

    return S;
}


VectorXd steady_state(MatrixXd const& S, VectorXd const& i) {
    MatrixXd I = MatrixXd::Identity(S.rows(), S.cols());
    return (I - S).inverse() * i;
}

VectorXd solve_minres(MatrixXd const& A, VectorXd const& b, size_t r) {
    int n = A.rows();    
    // first requirement for minres: matrix has to be symmetric so A=A^T 
    // our scattering matrix is tridiagonal and symmetric, photons scatter left and right with the same probability
    // second requirement (sort of): the matrix can be indefinite (unlike CG where the matrix has to be positive definite)
    // --> MINRES minimizes the norm of the residual (|| Ax - b ||), we don't have to calculate the inverse of the matrix
    // third requirement: matrix has to be regular for a solution A^-1 b = x
    MatrixXd Q = MatrixXd::Zero(n, r);

    // it's the krylov subspace
    // we create a krylow subscape of the dimension r (r directions in a n-)
    VectorXd v = b;
    for (size_t j = 0; j < r; ++j) {
        Q.col(j) = v;
        v = A * v; 
    }

    // HouseholderQR does a QR decomposition A = QR of a matrix A into a unitary matrix Q and upper triangular matrix R
    // this way our matrix has no entries below the diagonal
    // it mirrors a vector over a hyperplane onto an axis
    HouseholderQR<MatrixXd> QR(Q); // householder matrix is hermitian (has real number eigenvalues) [equal it's conjugate transpose]
    // given a complex vector w, the householder matrix is defined as such: U_w = Id - 2 (w^* . w)^-1 . w . w^*
    // the householder matrix also equals it's inverse matrix -> inverse = conjugate transpose therefore too
    // therefore eigenvalues are +- 1 ()
    // its determinant is: det(U_w) = -1
    // for a vector a we get: U_w . a = +- a  --> the householder matrix sets to zero all elements of a given vector except the first one
    // due to it being orthogonal and hermitian it doesn't change the vector's length
    
    // --> our Krylov subspace of vectors b, Ab, A^2b, ... creates sort of more parallel looking vectors, so still linear dependent
    // HouseholderQR creates an orthogonal set of vectors, makes it easier to find a solution
    

    MatrixXd Q_ortho = QR.householderQ() * MatrixXd::Identity(n, r);

    MatrixXd AQ = A * Q_ortho;
    VectorXd y = AQ.householderQr().solve(b); 
    // our best approximation is a linear combination of x ~ y0.b + y1.(Ab) + y2.(A^2b) + ...
    // it tries to solve these weights for which the residual is the lowest
    // the written-out version would somewhat look like this: argmin_y || y_0 * (Ab) + y_1 * (A^2 b) + y_2 * (A^3 b) ... - b ||,
    // meaning we are minimising the residual of the form || Ax - b ||

    // we must not forget to return to our space
    // this is basically getting x back, as y are only the basis coefficients: x = y_0 * q_0 + y_1 * q_1 + y_2*q_2
    // Q_ortho (n x r) * y (r x 1) to get a vector x of length n
    return Q_ortho * y;
}

int main() { 
    // some example values
    int n = 100;
    double kappa = 0.1;
    // size_t for size/length of arrays
    size_t r = 20;

    MatrixXd S = make_scattering_matrix(n, kappa);
    MatrixXd Other_matrix = MatrixXd::Identity(n, n) - S;

    // incoming current from the left end
    VectorXd i = VectorXd::Zero(n);
    i(0) = 1.0;

    VectorXd p = steady_state(S, i);
    VectorXd p_min = solve_minres(Other_matrix, i, r);

    // print first few values instead of tedious debugging
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

    // an ok to attempt at creating gnuplot
    ofstream gp("plot_steady_state.gnuplot");
    gp << "set terminal pdfcairo\n";
    gp << "set output 'steady_state.pdf'\n";
    gp << "set xlabel 'site index'\n";
    gp << "set ylabel 'steady state photons'\n";
    gp << "set title 'Steady state of scattering model'\n";
    gp << "plot 'steady_state.dat' using 1:2 with lines lw 2 title 'Reference', \\\n";
    gp << "     'steady_state.dat' u 1:3 w lines dt 2 lw 3 lc rgb '#FF0033' title 'MINRES (r=" << r << ")'\n";
    gp.close();

    cout << "\nData written to steady_state.dat\n";
    cout << "Gnuplot script written to plot_steady_state.gnuplot\n";
    cout << "Run: gnuplot plot_steady_state.gnuplot\n";

    return 0;
}