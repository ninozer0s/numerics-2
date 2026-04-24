// galaxy.cxx

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>      
#include <iomanip>
#include <stdexcept>
#include <vector>       
#include <cmath>        
#include <algorithm>    
#include <cctype>       

using namespace Eigen;
using namespace std;

constexpr double PI = 3.14159265358979323846; // NEW

// NEW
double deg2rad(double deg) {
    return deg * PI / 180.0;
}

// CHANGED: mit Kommas 
MatrixXd read_ecsv(ifstream& file, size_t cols) {
    string line;
    vector<vector<double>> rows;

    while (getline(file, line)) {
        if (line.empty()) continue;

        // skp headers
        if (!isdigit(line[0]) && line[0] != '-' && line[0] != '+') continue;

        // NEW: exchange commas to leerzeichen
        replace(line.begin(), line.end(), ',', ' ');

        istringstream stream(line);
        vector<double> row(cols);

        for (size_t j = 0; j < cols; ++j) {
            if (!(stream >> row[j])) {
                throw runtime_error("Could not parse ECSV data row.");
            }
        }

        rows.push_back(row);
    }

    if (rows.empty()) {
        throw runtime_error("No data rows found in ECSV file.");
    }

    MatrixXd values(rows.size(), cols);

    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            values(i, j) = rows[i][j];
        }
    }

    return values;
}

// NEW: Gaia-Daten ra, dec, parallax -> 3D-Koordinaten
MatrixXd gaia_to_cartesian(MatrixXd const& data) {
    const size_t n = data.rows();

    MatrixXd A(3, n);

    for (size_t i = 0; i < n; ++i) {
        double ra_deg = data(i, 0);
        double dec_deg = data(i, 1);
        double parallax = data(i, 2);

        if (parallax <= 0.0) {
            throw runtime_error("Parallax must be positive.");
        }

        double ra = deg2rad(ra_deg);
        double dec = deg2rad(dec_deg);

        // Parallax in milliarcseconds -> distance in parsec
        double r = 1000.0 / parallax;

        double x = r * cos(dec) * cos(ra);
        double y = r * cos(dec) * sin(ra);
        double z = r * sin(dec);

        A.col(i) << x, y, z;
    }

    return A;
}

// NEW: check Offdiagonal-Norm 
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

// CHANGED: QR-basierte SVD für A über Eigenzerlegung von A*A^T
size_t svd_qr(
    MatrixXd const& A, MatrixXd& U, VectorXd& s, MatrixXd& V,
    double max_off_diagonal, size_t max_iterations = 100
) {
    MatrixXd B = A * A.transpose();   // NEW: symm 3x3 Matrix
    MatrixXd Q_total = MatrixXd::Identity(B.rows(), B.cols());

    size_t iter = 0;

    for (; iter < max_iterations; ++iter) {
        HouseholderQR<MatrixXd> qr(B);

        MatrixXd Q = qr.householderQ();
        MatrixXd R = qr.matrixQR().template triangularView<Upper>();

        B = R * Q;
        Q_total = Q_total * Q;

        if (max_offdiag(B) < max_off_diagonal) {
            break;
        }
    }

    // Eigenwerte von A*A^T liegen ungefähr auf der Diagonale
    VectorXd eigenvalues = B.diagonal();

    // NEW: Singularwerte = sqrt(Eigenwerte)
    s.resize(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); ++i) {
        s(i) = sqrt(max(0.0, eigenvalues(i)));
    }

    U = Q_total;

    // NEW: nach absteigenden Singularwerten sortieren
    vector<int> idx(s.size());
    for (int i = 0; i < s.size(); ++i) idx[i] = i;

    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return s(a) > s(b);
    });

    VectorXd s_sorted(s.size());
    MatrixXd U_sorted(U.rows(), U.cols());

    for (int i = 0; i < s.size(); ++i) {
        s_sorted(i) = s(idx[i]);
        U_sorted.col(i) = U.col(idx[i]);
    }

    s = s_sorted;
    U = U_sorted;

    // NEW: V aus v_i = A^T u_i / sigma_i
    V = MatrixXd::Zero(A.cols(), s.size());

    for (int i = 0; i < s.size(); ++i) {
        if (s(i) > 1e-14) {
            V.col(i) = A.transpose() * U.col(i) / s(i);
        }
    }

    return iter + 1;
}

int main() {
    try {
        ifstream file("gaia_query.ecsv");

        if (!file) {
            throw runtime_error("Could not open ../gaia_query.ecsv");
        }

        // CHANGED: read gaia file
        MatrixXd data = read_ecsv(file, 5);

        cout << "Read stars: " << data.rows() << endl;

        // NEW: ra, dec, parallax -> kartesische Koordinaten
        MatrixXd A = gaia_to_cartesian(data);

        // NEW: Daten zentrieren
        Vector3d mean = A.rowwise().mean();
        A.colwise() -= mean;

        MatrixXd U, V;
        VectorXd s;

        // CHANGED: SVD auf 3 x n Koordinatenmatrix, nicht auf Rohdaten
        size_t iterations = svd_qr(A, U, s, V, 1e-10, 1000);

        cout << fixed << setprecision(10);

        cout << "QR iterations: " << iterations << endl;
        cout << "Singular values:\n" << s.transpose() << endl;

        // NEW: Normalvektor = U-Spalte zum kleinsten Singularwert
        Vector3d normal = U.col(2);
        normal.normalize();

        cout << "Plane normal vector:\n";
        cout << normal.transpose() << endl;

        // NEW: Referenz: North Galactic Pole, J2000
        // RA = 192.85948 deg, Dec = 27.12825 deg
        // Diese Richtung ist der Normalvektor der galaktischen Ebene.
        double ra_g = deg2rad(192.85948);
        double dec_g = deg2rad(27.12825);

        Vector3d galactic_normal;
        galactic_normal << cos(dec_g) * cos(ra_g),
                           cos(dec_g) * sin(ra_g),
                           sin(dec_g);
        galactic_normal.normalize();

        // Vorzeichen egal: n und -n beschreiben dieselbe Ebene
        double angle = acos(min(1.0, abs(normal.dot(galactic_normal)))) * 180.0 / PI;

        cout << "Reference galactic normal:\n";
        cout << galactic_normal.transpose() << endl;

        cout << "Angle to galactic plane normal [deg]: " << angle << endl;

    } catch (exception const& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}