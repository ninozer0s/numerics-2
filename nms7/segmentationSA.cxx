#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

using namespace std;

const int Nimax = 254;
const int Njmax = 333;
const int Nclass = 5;

// Reihenfolge: BG, WM, GM, CSF, SB
const string NAME[Nclass] = {"BG", "WM", "GM", "CSF", "SB"};
const double MW[Nclass]    = {30.0, 426.0, 602.0, 1223.0, 167.0};
const double SIGMA[Nclass] = {30.0,  59.0, 102.0,  307.0,  69.0};

// Parameter fuer Simulated Annealing
const double Ti = 3.0;
const double Tf = 0.02;
const double lambda = 1.05;
const int sweeps_per_T = 5;
const double J_SA = 1.2;
const unsigned seed = 123;

// Bayes-Energieterm:
// E_z(sigma) = (z - MW_sigma)^2/(2 Sigma_sigma^2) + ln(Sigma_sigma)
double local_energy(int z, int sigma) {
    const double dz = z - MW[sigma];
    return dz * dz / (2.0 * SIGMA[sigma] * SIGMA[sigma]) + log(SIGMA[sigma]);
}

void read_image(const string &filename, int image[Nimax][Njmax]) {
    ifstream in(filename.c_str());
    if (!in) {
        cerr << "Kann Datei nicht oeffnen: " << filename << endl;
        exit(1);
    }

    int ii, jj, value;
    for (int i = 0; i < Nimax; ++i) {
        for (int j = 0; j < Njmax; ++j) {
            in >> ii >> jj >> value;
            image[i][j] = value;
        }
    }
}

void write_segmentation(const string &filename, int seg[Nimax][Njmax]) {
    ofstream out(filename.c_str());
    if (!out) {
        cerr << "Kann Datei nicht schreiben: " << filename << endl;
        exit(1);
    }

    for (int i = 0; i < Nimax; ++i) {
        for (int j = 0; j < Njmax; ++j) {
            out << i + 1 << " " << j + 1 << " " << seg[i][j] + 1 << "\n";
        }
        out << "\n";
    }
}

void local_segmentation(int MR[Nimax][Njmax], int seg[Nimax][Njmax]) {
    for (int i = 0; i < Nimax; ++i) {
        for (int j = 0; j < Njmax; ++j) {
            int best_sigma = 0;
            double best_E = local_energy(MR[i][j], 0);

            for (int sigma = 1; sigma < Nclass; ++sigma) {
                const double E = local_energy(MR[i][j], sigma);
                if (E < best_E) {
                    best_E = E;
                    best_sigma = sigma;
                }
            }

            seg[i][j] = best_sigma;
        }
    }
}

int count_equal_neighbours(int seg[Nimax][Njmax], int i, int j, int sigma) {
    int n = 0;

    if (i > 0        && seg[i - 1][j] == sigma) ++n;
    if (i < Nimax-1 && seg[i + 1][j] == sigma) ++n;
    if (j > 0        && seg[i][j - 1] == sigma) ++n;
    if (j < Njmax-1 && seg[i][j + 1] == sigma) ++n;

    return n;
}

void simulated_annealing(int MR[Nimax][Njmax], int seg[Nimax][Njmax]) {
    // Start mit der lokalen J=0-Loesung.
    local_segmentation(MR, seg);

    mt19937 rng(seed);
    uniform_int_distribution<int> random_i(0, Nimax - 1);
    uniform_int_distribution<int> random_j(0, Njmax - 1);
    uniform_int_distribution<int> random_sigma(0, Nclass - 1);
    uniform_real_distribution<double> random01(0.0, 1.0);

    const int Npixel = Nimax * Njmax;

    for (double T = Ti; T > Tf; T /= lambda) {
        for (int sweep = 0; sweep < sweeps_per_T; ++sweep) {
            for (int step = 0; step < Npixel; ++step) {
                const int i = random_i(rng);
                const int j = random_j(rng);

                const int old_sigma = seg[i][j];

                int new_sigma = random_sigma(rng);
                if (new_sigma == old_sigma) {
                    new_sigma = (new_sigma + 1) % Nclass;
                }

                // lokale Energieaenderung
                const double dE_local =
                    local_energy(MR[i][j], new_sigma)
                    - local_energy(MR[i][j], old_sigma);

                // Nachbarschaftsenergie fuer 4 naechste Nachbarn
                const int same_new = count_equal_neighbours(seg, i, j, new_sigma);
                const int same_old = count_equal_neighbours(seg, i, j, old_sigma);

                const double dE_neigh = -J_SA * (same_new - same_old);
                const double dE = dE_local + dE_neigh;

                // Metropolis-Regel
                if (dE <= 0.0 || random01(rng) < exp(-dE / T)) {
                    seg[i][j] = new_sigma;
                }
            }
        }
    }
}

struct ErrorInfo {
    long wrong[Nclass] = {0, 0, 0, 0, 0};
    long total[Nclass] = {0, 0, 0, 0, 0};
};

ErrorInfo compute_errors(int seg[Nimax][Njmax], int correct[Nimax][Njmax]) {
    ErrorInfo e;

    for (int i = 0; i < Nimax; ++i) {
        for (int j = 0; j < Njmax; ++j) {
            const int c = correct[i][j] - 1;

            ++e.total[c];

            if (seg[i][j] != c) {
                ++e.wrong[c];
            }
        }
    }

    return e;
}

void print_errors(ofstream &out, const string &title, const ErrorInfo &e) {
    long wrong_all = 0;
    long total_all = 0;

    out << title << "\n";
    out << "Gewebe  falsche Pixel / Gesamtpixel  Fehler\n";

    for (int s = 0; s < Nclass; ++s) {
        wrong_all += e.wrong[s];
        total_all += e.total[s];

        out << setw(4) << NAME[s] << "   "
            << setw(6) << e.wrong[s] << " / "
            << setw(6) << e.total[s]
            << "        "
            << fixed << setprecision(6)
            << double(e.wrong[s]) / double(e.total[s])
            << "\n";
    }

    out << "Gesamt " << wrong_all << " / " << total_all
        << "        "
        << fixed << setprecision(6)
        << double(wrong_all) / double(total_all)
        << "\n\n";
}

void write_protocol(const ErrorInfo &err_local, const ErrorInfo &err_sa) {
    ofstream out("Protokoll.txt");
    if (!out) {
        cerr << "Kann Protokoll.txt nicht schreiben.\n";
        exit(1);
    }

    out << "Simulated-Annealing von NMR-Gehirn-Bildern\n";
    out << "============================================\n\n";

    out << "Modell:\n";
    out << "H = -J Summe_<ij> delta_{sigma_i,sigma_j} + Summe_i E_{z_i}(sigma_i)\n";
    out << "E_z(sigma) = (z - MW_sigma)^2/(2 SIGMA_sigma^2) + ln(SIGMA_sigma)\n";
    out << "Es werden die 4 naechsten Nachbarn verwendet.\n\n";

    out << "Parameterwahl:\n";
    out << "Ti = " << Ti << ": hoch genug, damit am Anfang auch schlechtere lokale Schritte akzeptiert werden.\n";
    out << "Tf = " << Tf << ": klein genug, damit am Ende praktisch nur noch energiesenkende Schritte akzeptiert werden.\n";
    out << "lambda = " << lambda << ": langsames geometrisches Abkuehlen T -> T/lambda.\n";
    out << "# sweeps = " << sweeps_per_T << " pro Temperatur: Kompromiss aus Laufzeit und Relaxation.\n";
    out << "J = " << J_SA << ": Nachbarschaftskorrelation glaettet das Bild, ohne kleine Gewebe komplett zu unterdruecken.\n";
    out << "Zufallsseed = " << seed << ": feste Wahl fuer reproduzierbare Ergebnisse.\n\n";

    print_errors(out, "Vergleichsrechnung J = 0 (SegLocal.dat)", err_local);
    print_errors(out, "Simulated Annealing mit J != 0 (SegSA.dat)", err_sa);

    out << "Erzeugte Dateien:\n";
    out << "SegLocal.dat, SegSA.dat, Protokoll.txt\n";
    out << "PS-Dateien werden mit plotSeg.gnu erzeugt.\n";
}

int main() {
    static int MR[Nimax][Njmax];
    static int correct[Nimax][Njmax];
    static int seg_local[Nimax][Njmax];
    static int seg_sa[Nimax][Njmax];

    read_image("SimMRimage.dat", MR);
    read_image("CorrectSegImage.dat", correct);

    local_segmentation(MR, seg_local);       // J = 0
    simulated_annealing(MR, seg_sa);         // J != 0

    write_segmentation("SegLocal.dat", seg_local);
    write_segmentation("SegSA.dat", seg_sa);

    const ErrorInfo err_local = compute_errors(seg_local, correct);
    const ErrorInfo err_sa = compute_errors(seg_sa, correct);

    write_protocol(err_local, err_sa);

    cout << "Fertig.\n";
    cout << "Erzeugt: SegLocal.dat, SegSA.dat, Protokoll.txt\n";

    return 0;
}