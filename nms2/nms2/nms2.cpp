// nms2.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace Eigen;
using namespace std;


MatrixXd read_ecsv(ifstream& file, size_t cols) {
    // ecsv contains data in equatorial coordinate system with the right ascension and declination
    // right ascension is the angular distance horizontally
    // declination is the vertical angular distance
    // the origin lies at the center of the earth
    string line;
    double ra, dec, parallax, parallax_over_error, phot_g_mean_mag;
    vector < vector < double > > rows;

    while (getline(file, line)) {
        if (!isdigit(line[0]) && line[0] != '-') continue;
        istringstream stream(line);
        // read elements and assemble in rows vector
        stream >> ra >> dec >> parallax >> parallax_over_error >> phot_g_mean_mag;
        rows.push_back({ ra, dec, parallax, parallax_over_error, phot_g_mean_mag });
    }

    // convert vector of vectors to MatrixXd and return it
    MatrixXd gaia_values(rows.size(), rows[0].size());

    for (size_t i = 0; i < rows.size(); i++) {
        for (size_t j = 0; j < rows[0].size(); j++) {
            gaia_values(i, j) = rows[i][j];
        }
    }
    
    return gaia_values;
}

size_t svd_qr(
    MatrixXd const& A, MatrixXd& U, VectorXd& s, MatrixXd& V,
    double max_off_diagonal, size_t max_iterations = 100
) {
    // compute SVD and return number of iterations needed
    // such that: | S^ i_j | < max_off_diagonal for all i != j
    MatrixXd Ak = A;
    size_t i = 0;
    for (; i < max_iterations; i++) {
        HouseholderQR<MatrixXd> qr(Ak);
        MatrixXd Q = qr.householderQ();
        MatrixXd R = qr.matrixQR().template triangularView<Upper>();;
        
        Ak = R * Q;
    }
    return i;
}

int main()
{
    ifstream file("../gaia_query.ecsv");
    size_t cols = 5;
    MatrixXd babe = read_ecsv(file, cols);

    MatrixXd U, V;
    VectorXd s;
    size_t iterations = svd_qr(babe, U, s, V, 1e-10);
    cout << iterations;
}