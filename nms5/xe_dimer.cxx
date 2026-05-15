#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

/*******************************************************************/
// declare functions
/*******************************************************************/
// Numerow functions
void integrate_y_using_numerow(VectorXd &y, VectorXd &g, VectorXd &s, size_t nr, double dr);

// Functions needed to solve Schrödinger equation
void solve_se(size_t tnodes, double &emin, double &emax, VectorXd &pot, VectorXd &r, size_t nr, double dr, VectorXd &psi);
void calc_g(double e, VectorXd &pot, VectorXd &r, size_t nr, VectorXd &g);
void normalize_psi(VectorXd &psi, VectorXd &r, size_t nr, double dr);
void init_psi_for_se(VectorXd &psi, VectorXd &r, size_t nr);
size_t calc_nodes(VectorXd &psi, size_t nr);
void init_pot(VectorXd &pot, VectorXd &r, size_t nr, double dr);

int main() {
   size_t nr(2000);
   double rmax(30.0);
   double rmin(6.0);
   double dr;
   dr = abs(rmax - rmin) / (nr - 1);

   VectorXd r(nr);
   for (size_t i = 0; i < nr; ++i) {
      r(i) = rmin + i * (rmax - rmin) / (nr - 1);
   }

   VectorXd psi(nr);
   VectorXd pot(nr);

   size_t nnodes(16);
   VectorXd energy(nnodes);

   VectorXd exp_delta_en(17);
   exp_delta_en(0) = 19.90;
   exp_delta_en(1) = 18.55;
   exp_delta_en(2) = 17.20;
   exp_delta_en(3) = 16.17;
   exp_delta_en(4) = 14.63;
   exp_delta_en(5) = 13.70;
   exp_delta_en(6) = 12.63;
   exp_delta_en(7) = 11.33;
   exp_delta_en(8) = 10.15;
   exp_delta_en(9) = 8.95;
   exp_delta_en(10) = 7.83;
   exp_delta_en(11) = 6.79;
   exp_delta_en(12) = 5.83;
   exp_delta_en(13) = 4.93;
   exp_delta_en(14) = 4.11;
   exp_delta_en(15) = 3.37;
   exp_delta_en(16) = 2.70;

   init_pot(pot, r, nr, dr);

   ofstream outpot("pot.dat");
   streambuf *coutbuf = cout.rdbuf();
   cout.rdbuf(outpot.rdbuf());
   for (size_t i = 0; i < nr; ++i) {
      cout << r(i) << " " << setprecision(16) << pot(i) << " " << endl;
   }
   cout.rdbuf(coutbuf);
   outpot.close();

   cout << "Welcome to the SE-Solver for the Xe dimer" << endl;

   double emax, emin;

   for (size_t i = 0; i < nnodes; ++i) {
      emin = pot.minCoeff();
      emax = 0.0;

      solve_se(i, emin, emax, pot, r, nr, dr, psi);

      cout << "For " << i << " nodes. Lowest EV is between "
           << emin << " and " << emax << " Hartree." << endl;

      energy(i) = 0.5 * (emin + emax);

      cout << "Writing computed wavefunction to psi" + to_string(i) + ".dat" << endl;

      ofstream outpsi("psi" + to_string(i) + ".dat");
      cout.rdbuf(outpsi.rdbuf());
      for (size_t j = 0; j < nr; ++j) {
         cout << r(j) << " " << setprecision(16) << psi(j) << " " << endl;
      }
      cout.rdbuf(coutbuf);
      outpsi.close();
   }

   cout << endl;
   cout << "==================================================================================" << endl;
   cout << "Final computed energy differences for the 15 lowest vibrational modes of Xe-dimer." << endl;
   cout << "==================================================================================" << endl;
   cout << "    E(v+1)-E(v) " << endl;
   cout << "v  Theory  Experiment " << endl;
   cout << "==================================================================================" << endl;

   const double hartree_to_cm = 219474.6313705;

   for (size_t i = 0; i < nnodes - 1; ++i) {
      double delta_e_cm = (energy(i + 1) - energy(i)) * hartree_to_cm;
      cout << i << "      " << fixed << setprecision(2)
           << delta_e_cm << "     " << exp_delta_en(i) << endl;
   }

   cout << endl;

   // convergence check with different grid spacings
   vector<size_t> convergence_grids = {1000, 2000, 4000};
   ofstream convout("convergence.dat");

   convout << "# nr dr E0_cm E1_cm E2_cm E3_cm E4_cm E5_cm E6_cm E7_cm "
           << "E8_cm E9_cm E10_cm E11_cm E12_cm E13_cm E14_cm E15_cm" << endl;

   for (size_t igrid = 0; igrid < convergence_grids.size(); ++igrid) {
      size_t nr_conv = convergence_grids[igrid];
      double dr_conv = abs(rmax - rmin) / (nr_conv - 1);

      VectorXd r_conv(nr_conv);
      VectorXd pot_conv(nr_conv);
      VectorXd psi_conv(nr_conv);
      VectorXd energy_conv(nnodes);

      for (size_t j = 0; j < nr_conv; ++j) {
         r_conv(j) = rmin + j * (rmax - rmin) / (nr_conv - 1);
      }

      init_pot(pot_conv, r_conv, nr_conv, dr_conv);

      for (size_t inode = 0; inode < nnodes; ++inode) {
         double emin_conv = pot_conv.minCoeff();
         double emax_conv = 0.0;

         solve_se(inode, emin_conv, emax_conv, pot_conv, r_conv, nr_conv, dr_conv, psi_conv);

         energy_conv(inode) = 0.5 * (emin_conv + emax_conv);
      }

      convout << nr_conv << " " << setprecision(12) << dr_conv;

      for (size_t inode = 0; inode < nnodes; ++inode) {
         convout << " " << setprecision(12) << energy_conv(inode) * hartree_to_cm;
      }

      convout << endl;
   }

   convout.close();

   // Plot 1: only ground-state wavefunction
   ofstream gp1("plot_groundstate.gp");
   gp1 << "set terminal pngcairo size 900,600\n";
   gp1 << "set output 'psi0.png'\n";
   gp1 << "set xlabel 'r / bohr'\n";
   gp1 << "set ylabel 'psi_0(r)'\n";
   gp1 << "set xrange [7:10]\n";
   gp1 << "set title 'Ground-state wavefunction around potential minimum'\n";
   gp1 << "plot 'psi0.dat' using 1:2 with lines title 'psi_0'\n";
   gp1.close();

   // Plot 2: potential and scaled ground-state wavefunction
   ofstream gp2("plot_potential_and_groundstate.gp");
   gp2 << "set terminal pngcairo size 900,600\n";
   gp2 << "set output 'potential_and_psi0.png'\n";
   gp2 << "set xlabel 'r / bohr'\n";
   gp2 << "set xrange [7:10]\n";
   gp2 << "set title 'Ground-state wavefunction around potential minimum'\n";
   gp2 << "set ylabel 'V(r) / Hartree'\n";
   gp2 << "set y2label 'scaled psi_0(r)'\n";
   gp2 << "set y2tics\n";
   gp2 << "maxpsi = system(\"awk 'BEGIN{m=0} {a=$2; if(a<0)a=-a; if(a>m)m=a} END{print m}' psi0.dat\")\n";
   gp2 << "plot 'pot.dat' using 1:2 with lines axis x1y1 title 'V(r)', \\\n";
   gp2 << "     'psi0.dat' using 1:($2/maxpsi) with lines axis x1y2 title 'scaled psi_0'\n";
   gp2.close();

   // Automatically create png files if gnuplot is installed
   system("gnuplot plot_groundstate.gp");
   system("gnuplot plot_potential_and_groundstate.gp");

   return 0;
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//! Applying Numerow method to solve
//!
//! d^2 Y(x)
//! -------- = -G(x) * Y(x) + S(x)
//!  dx^2
//!
//! by integration from inside to outside.
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void integrate_y_using_numerow(VectorXd &y, VectorXd &g, VectorXd &s, size_t nr, double dr) {
   const double dr2 = dr * dr;

   for (size_t i = 1; i < (nr - 1); ++i) {
      y(i + 1) =
         (
            2.0 * y(i) * (1.0 - 5.0 / 12.0 * dr2 * g(i))
            - y(i - 1) * (1.0 + dr2 / 12.0 * g(i - 1))
            + dr2 / 12.0 * (s(i + 1) + 10.0 * s(i) + s(i - 1))
         )
         /
         (1.0 + dr2 / 12.0 * g(i + 1));

      // Avoid overflow during shooting.
      if (abs(y(i + 1)) > 1.0e100) {
         y *= 1.0e-100;
      }
   }
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//! Main function to solve the Schroedinger equation
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void solve_se(size_t tnodes, double &emin, double &emax, VectorXd &pot, VectorXd &r, size_t nr, double dr, VectorXd &psi) {
   VectorXd g(nr);
   VectorXd s(nr);
   s.setZero();

   // Tolerance in Hartree.
   // 1 cm^-1 is about 4.56e-6 Hartree, so this is much stricter.
   double tol(1.0e-10);

   auto shoot = [&](double e, VectorXd *psi_out, size_t *nodes_out) -> double {
      VectorXd y(nr);

      calc_g(e, pot, r, nr, g);
      init_psi_for_se(y, r, nr);
      integrate_y_using_numerow(y, g, s, nr, dr);

      if (nodes_out != nullptr) {
         *nodes_out = calc_nodes(y, nr);
      }

      if (psi_out != nullptr) {
         *psi_out = y;
      }

      return y(nr - 1);
   };

   double scan_min = emin + 1.0e-12;
   double scan_max = emax - 1.0e-12;

   const size_t nscan = 5000;

   double e_left = scan_min;
   double f_left = shoot(e_left, nullptr, nullptr);

   bool bracket_found = false;
   size_t root_counter = 0;

   // Find the tnodes-th eigenvalue by scanning for sign changes.
   for (size_t iscan = 1; iscan <= nscan; ++iscan) {
      double e_right =
         scan_min + (scan_max - scan_min) * static_cast<double>(iscan) / static_cast<double>(nscan);

      double f_right = shoot(e_right, nullptr, nullptr);

      if ((f_left <= 0.0 && f_right >= 0.0) ||
          (f_left >= 0.0 && f_right <= 0.0)) {

         if (root_counter == tnodes) {
            emin = e_left;
            emax = e_right;
            bracket_found = true;
            break;
         }

         ++root_counter;
      }

      e_left = e_right;
      f_left = f_right;
   }

   if (!bracket_found) {
      throw runtime_error("No eigenvalue bracket found. Increase rmax/nscan or check the energy interval.");
   }

   double f_emin = shoot(emin, nullptr, nullptr);

   // Bisection inside the bracket.
   while (abs(emax - emin) > tol) {
      double e = 0.5 * (emin + emax);

      size_t nodes = 0;
      double f_mid = shoot(e, nullptr, &nodes);

      if ((f_emin <= 0.0 && f_mid >= 0.0) ||
          (f_emin >= 0.0 && f_mid <= 0.0)) {
         emax = e;
      } else {
         emin = e;
         f_emin = f_mid;
      }
   }

   double e = 0.5 * (emin + emax);

   size_t nodes = 0;
   shoot(e, &psi, &nodes);

   // Choose positive phase convention.
   if (psi(nr / 2) < 0.0) {
      psi *= -1.0;
   }

   normalize_psi(psi, r, nr, dr);
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//! Calculate g(r) for Numerow
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void calc_g(double e, VectorXd &pot, VectorXd &r, size_t nr, VectorXd &g) {
   double MXe(240446.56);
   double Mred(MXe / 2.0);

   for (size_t i = 0; i < nr; ++i) {
      g(i) = 2.0 * Mred * (e - pot(i));
   }
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//! Initial values for psi
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void init_psi_for_se(VectorXd &psi, VectorXd &r, size_t nr) {
   psi.setZero();

   // Boundary condition at small r: wavefunction is approximately zero.
   // The second value only starts the recurrence.
   psi(0) = 0.0;
   psi(1) = 1.0e-12;
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//! Normalize psi
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void normalize_psi(VectorXd &psi, VectorXd &r, size_t nr, double dr) {
   // First rescale by max value to avoid underflow/overflow in the norm.
   double max_abs = 0.0;

   for (size_t i = 0; i < nr; ++i) {
      if (abs(psi(i)) > max_abs) {
         max_abs = abs(psi(i));
      }
   }

   if (max_abs == 0.0) {
      return;
   }

   psi /= max_abs;

   double norm = 0.0;

   // Trapezoidal integration of |psi|^2.
   for (size_t i = 0; i < nr; ++i) {
      double weight = 1.0;

      if (i == 0 || i == nr - 1) {
         weight = 0.5;
      }

      norm += weight * psi(i) * psi(i);
   }

   norm *= dr;

   if (norm > 0.0) {
      psi /= sqrt(norm);
   }
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//! Count nodes of psi
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

size_t calc_nodes(VectorXd &psi, size_t nr) {
   size_t nodes(0);

   int previous_sign = 0;

   for (size_t i = 0; i < nr; ++i) {
      int current_sign = 0;

      if (psi(i) > 0.0) {
         current_sign = 1;
      }

      if (psi(i) < 0.0) {
         current_sign = -1;
      }

      if (current_sign != 0) {
         if (previous_sign != 0 && current_sign != previous_sign) {
            ++nodes;
         }

         previous_sign = current_sign;
      }
   }

   return nodes;
}


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//! Initialize Xe-dimer potential
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void init_pot(VectorXd &pot, VectorXd &r, size_t nr, double dr) {
   size_t nc(12);

   VectorXd exp(nc);
   VectorXd coeff(nc);

   exp(0) = 6;
   exp(1) = 8;
   exp(2) = 9;
   exp(3) = 10;
   exp(4) = 11;
   exp(5) = 12;
   exp(6) = 13;
   exp(7) = 14;
   exp(8) = 15;
   exp(9) = 16;
   exp(10) = 17;
   exp(11) = 18;

   coeff(0) = -301.7;
   coeff(1) = -26816.402071;
   coeff(2) = -29141425.4118978;
   coeff(3) = 2525729440.60837;
   coeff(4) = -93157553751.1815;
   coeff(5) = 1958061699137.66;
   coeff(6) = -25959609531187.9;
   coeff(7) = 225015902487099;
   coeff(8) = -1272921381781360;
   coeff(9) = 4526862108942420;
   coeff(10) = -9182845674164360;
   coeff(11) = 8100817151233585;

   pot.setZero();

   for (size_t i = 0; i < nr; ++i) {
      for (size_t j = 0; j < nc; ++j) {
         pot(i) += coeff(j) / pow(r(i), exp(j));
      }
   }
}