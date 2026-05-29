#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

struct Result {
    int L;
    double T;
    double J;
    double E_per_spin;
    double m_avg;
    double abs_m_avg;
    double m2_avg;
    double m4_avg;
    double chi;
    double binder;
    double acceptance;
};

int pbc(int i, int L) {
    if (i < 0) return L - 1;
    if (i >= L) return 0;
    return i;
}

double magnetization_per_spin(const vector<int>& s) {
    long long M = 0;

    for (int spin : s) {
        M += spin;
    }

    return double(M) / double(s.size());
}

double energy_per_spin(const vector<int>& s, int L, double J, double B) {
    double E = 0.0;

    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            int i = x * L + y;
            int spin = s[i];

            int right = s[x * L + pbc(y + 1, L)];
            int down  = s[pbc(x + 1, L) * L + y];

            // every bond is counted once: right and down neighbor only
            E += -J * spin * (right + down);

            // magnetic field term
            E += -B * spin;
        }
    }

    return E / double(L * L);
}

void initialize_spins(vector<int>& s, mt19937& rng, bool ordered) {
    if (ordered) {
        for (int& spin : s) {
            spin = 1;
        }
    } else {
        uniform_real_distribution<double> uni(0.0, 1.0);

        for (int& spin : s) {
            spin = (uni(rng) < 0.5) ? -1 : 1;
        }
    }
}

bool metropolis_flip(vector<int>& s,
                     int L,
                     double T,
                     double J,
                     double B,
                     mt19937& rng,
                     uniform_real_distribution<double>& uni,
                     uniform_int_distribution<int>& site_dist) {
    int N = L * L;

    int i = site_dist(rng);
    int x = i / L;
    int y = i % L;

    int spin = s[i];

    int nn_sum =
        s[pbc(x + 1, L) * L + y] +
        s[pbc(x - 1, L) * L + y] +
        s[x * L + pbc(y + 1, L)] +
        s[x * L + pbc(y - 1, L)];

    // energy difference for spin flip spin -> -spin
    // H = -J sum_<ij> s_i s_j - B sum_i s_i
    double dE = 2.0 * spin * (J * nn_sum + B);

    // Metropolis acceptance rule
    if (dE <= 0.0 || uni(rng) < exp(-dE / T)) {
        s[i] = -spin;
        return true;
    }

    return false;
}

long long sweep(vector<int>& s,
                int L,
                double T,
                double J,
                double B,
                mt19937& rng,
                uniform_real_distribution<double>& uni,
                uniform_int_distribution<int>& site_dist) {
    int N = L * L;
    long long accepted = 0;

    // one sweep = N attempted spin flips
    for (int k = 0; k < N; ++k) {
        if (metropolis_flip(s, L, T, J, B, rng, uni, site_dist)) {
            accepted++;
        }
    }

    return accepted;
}

Result simulate(int L,
                double T,
                double J,
                double B,
                int warmup_sweeps,
                int measurement_sweeps,
                unsigned seed) {
    int N = L * L;

    mt19937 rng(seed);
    uniform_real_distribution<double> uni(0.0, 1.0);
    uniform_int_distribution<int> site_dist(0, N - 1);

    vector<int> spins(N);

    // ordered start is good for ferromagnetic low-temperature runs
    bool ordered_start = (J > 0.0);
    initialize_spins(spins, rng, ordered_start);

    // warm-up sweeps are not measured
    for (int w = 0; w < warmup_sweeps; ++w) {
        sweep(spins, L, T, J, B, rng, uni, site_dist);
    }

    double E_sum = 0.0;
    double m_sum = 0.0;
    double abs_m_sum = 0.0;
    double m2_sum = 0.0;
    double m4_sum = 0.0;

    long long accepted = 0;

    for (int sw = 0; sw < measurement_sweeps; ++sw) {
        accepted += sweep(spins, L, T, J, B, rng, uni, site_dist);

        double m = magnetization_per_spin(spins);
        double m2 = m * m;
        double m4 = m2 * m2;

        E_sum += energy_per_spin(spins, L, J, B);
        m_sum += m;
        abs_m_sum += fabs(m);
        m2_sum += m2;
        m4_sum += m4;
    }

    double nmeas = double(measurement_sweeps);

    Result r;

    r.L = L;
    r.T = T;
    r.J = J;
    r.E_per_spin = E_sum / nmeas;
    r.m_avg = m_sum / nmeas;
    r.abs_m_avg = abs_m_sum / nmeas;
    r.m2_avg = m2_sum / nmeas;
    r.m4_avg = m4_sum / nmeas;

    // susceptibility per spin, k_B = 1
    // for finite systems this version is more stable:
    // chi = N * (<m^2> - <|m|>^2) / T
    r.chi = double(N) * (r.m2_avg - r.abs_m_avg * r.abs_m_avg) / T;

    // Binder cumulant:
    // U4 = 1 - <m^4> / (3 <m^2>^2)
    r.binder = 1.0 - r.m4_avg / (3.0 * r.m2_avg * r.m2_avg);

    r.acceptance = double(accepted) / double(measurement_sweeps * N);

    return r;
}

vector<double> make_temperature_grid() {
    vector<double> T;

    // coarse grid, includes T = 1, 3, 5, 7
    for (double x = 1.0; x <= 7.0001; x += 0.5) {
        T.push_back(x);
    }

    // finer grid around critical temperature
    for (double x = 1.8; x <= 2.8001; x += 0.1) {
        T.push_back(x);
    }

    // very fine grid around Tc ~ 2.269
    for (double x = 2.15; x <= 2.4001; x += 0.025) {
        T.push_back(x);
    }

    // remove duplicates
    vector<double> clean;

    for (double x : T) {
        bool exists = false;

        for (double y : clean) {
            if (fabs(x - y) < 1e-9) {
                exists = true;
                break;
            }
        }

        if (!exists) {
            clean.push_back(x);
        }
    }

    // sort
    for (size_t i = 0; i < clean.size(); ++i) {
        for (size_t j = i + 1; j < clean.size(); ++j) {
            if (clean[j] < clean[i]) {
                double tmp = clean[i];
                clean[i] = clean[j];
                clean[j] = tmp;
            }
        }
    }

    return clean;
}

void write_gnuplot_script(const string& datafile, const string& scriptfile) {
    ofstream gp(scriptfile);

    gp << "set terminal pdfcairo enhanced color size 8,6\n";
    gp << "set output 'ising_plots.pdf'\n";
    gp << "set datafile commentschars '#'\n";
    gp << "set key outside\n";
    gp << "set grid\n";
    gp << "set xlabel 'T'\n";
    gp << "set xzeroaxis\n\n";

    // columns:
    // 1 L
    // 2 T
    // 3 J
    // 4 E_per_spin
    // 5 <m>
    // 6 <|m|>
    // 7 <m^2>
    // 8 <m^4>
    // 9 chi
    // 10 binder
    // 11 acceptance

    auto plot_block = [&](const string& ylabel, int col, const string& title) {
        gp << "set ylabel '" << ylabel << "'\n";
        gp << "set title '" << title << "'\n";
        gp << "plot \\\n";
        gp << "  '" << datafile << "' using 2:(($1==5)?$" << col << ":1/0) with linespoints title 'L=5', \\\n";
        gp << "  '" << datafile << "' using 2:(($1==10)?$" << col << ":1/0) with linespoints title 'L=10', \\\n";
        gp << "  '" << datafile << "' using 2:(($1==20)?$" << col << ":1/0) with linespoints title 'L=20', \\\n";
        gp << "  '" << datafile << "' using 2:(($1==40)?$" << col << ":1/0) with linespoints title 'L=40'\n\n";
    };

    plot_block("<|m|>", 6, "Absolute magnetization per spin");
    plot_block("<m^2>", 7, "Second moment of magnetization");
    plot_block("<m^4>", 8, "Fourth moment of magnetization");
    plot_block("chi", 9, "Magnetic susceptibility per spin");
    plot_block("U_4", 10, "Binder cumulant");

    gp << "set output\n";
}

double estimate_tc_from_binder(const vector<Result>& results) {
    // crude estimate from crossing of Binder curves for L=20 and L=40
    vector<Result> L20;
    vector<Result> L40;

    for (const auto& r : results) {
        if (r.L == 20) L20.push_back(r);
        if (r.L == 40) L40.push_back(r);
    }

    size_t n = min(L20.size(), L40.size());

    for (size_t i = 1; i < n; ++i) {
        double T0 = L20[i - 1].T;
        double T1 = L20[i].T;

        double d0 = L20[i - 1].binder - L40[i - 1].binder;
        double d1 = L20[i].binder - L40[i].binder;

        if (d0 == 0.0) {
            return T0;
        }

        if (d0 * d1 < 0.0) {
            // linear interpolation of crossing
            return T0 + (T1 - T0) * fabs(d0) / (fabs(d0) + fabs(d1));
        }
    }

    return -1.0;
}

void write_readme(int warmup_sweeps,
                  int measurement_sweeps,
                  double J,
                  double B,
                  double Tc_estimate) {
    ofstream readme("Readme.md");

    readme << "# Monte-Carlo-Simulation des 2D-Ising-Modells\n\n";

    readme << "## Methode\n\n";
    readme << "Simuliert wird das Ising-Modell auf einem quadratischen Gitter mit periodischen Randbedingungen.\n";
    readme << "Verwendet wird der Single-Spin-Flip-Metropolis-Algorithmus.\n\n";

    readme << "Hamiltonian:\n\n";
    readme << "`H = -J sum_<ij> s_i s_j - B sum_i s_i`\n\n";

    readme << "Ein Sweep besteht aus `L*L` versuchten Spin-Flips.\n\n";

    readme << "## Parameter\n\n";
    readme << "- J = " << J << "\n";
    readme << "- B = " << B << "\n";
    readme << "- Systemgroessen: L = 5, 10, 20, 40\n";
    readme << "- Warm-up sweeps: " << warmup_sweeps << "\n";
    readme << "- Measurement sweeps: " << measurement_sweeps << "\n";
    readme << "- Temperaturen: T = 1 bis 7, plus feines Raster um T = 2.269\n";
    readme << "- Zufallszahlengenerator: std::mt19937\n\n";

    readme << "## Gemessene Groessen\n\n";
    readme << "Die Magnetisierung wird pro Spin verwendet: m = M / (L*L).\n\n";
    readme << "- <|m|>\n";
    readme << "- <m^2>\n";
    readme << "- <m^4>\n";
    readme << "- Suszeptibilitaet: chi = N * (<m^2> - <|m|>^2) / T\n";
    readme << "- Binder-Kumulante: U4 = 1 - <m^4> / (3 <m^2>^2)\n\n";

    readme << "## Kritische Temperatur\n\n";

    if (Tc_estimate > 0.0) {
        readme << "Aus dem Schnittpunkt der Binder-Kumulanten fuer L=20 und L=40 ergibt sich grob:\n\n";
        readme << "`Tc approx " << Tc_estimate << "`\n\n";
    } else {
        readme << "Der Schnittpunkt der Binder-Kumulanten konnte mit dem aktuellen Raster nicht automatisch bestimmt werden.\n\n";
    }

    readme << "Der erwartete Wert fuer das 2D-Ising-Modell auf dem Quadratgitter bei J=1 und kB=1 ist ungefaehr:\n\n";
    readme << "`Tc approx 2.269`\n\n";

    readme << "## Output\n\n";
    readme << "- ising_results.dat: Messdaten\n";
    readme << "- ising_plots.pdf: Plots aller geforderten Groessen\n";
    readme << "- ising_plots.gp: automatisch erzeugtes gnuplot-Skript\n";
}

int main() {
    const double J = 1.0;
    const double B = 0.0;

    // for a quick test you can reduce these numbers
    const int warmup_sweeps = 5000;
    const int measurement_sweeps = 20000;

    vector<int> L_values = {5, 10, 20, 40};
    vector<double> T_values = make_temperature_grid();

    vector<Result> results;

    ofstream out("ising_results.dat");

    out << "# columns:\n";
    out << "# 1 L\n";
    out << "# 2 T\n";
    out << "# 3 J\n";
    out << "# 4 E_per_spin\n";
    out << "# 5 <m>\n";
    out << "# 6 <|m|>\n";
    out << "# 7 <m^2>\n";
    out << "# 8 <m^4>\n";
    out << "# 9 chi\n";
    out << "# 10 binder\n";
    out << "# 11 acceptance\n";

    cout << "Starting simulation...\n";
    cout << "This can take a while, especially for L=40.\n\n";

    unsigned base_seed = 12345;

    for (int L : L_values) {
        for (size_t it = 0; it < T_values.size(); ++it) {
            double T = T_values[it];

            unsigned seed = base_seed + 1000 * L + unsigned(it);

            Result r = simulate(
                L,
                T,
                J,
                B,
                warmup_sweeps,
                measurement_sweeps,
                seed
            );

            results.push_back(r);

            out << fixed << setprecision(10)
                << r.L << " "
                << r.T << " "
                << r.J << " "
                << r.E_per_spin << " "
                << r.m_avg << " "
                << r.abs_m_avg << " "
                << r.m2_avg << " "
                << r.m4_avg << " "
                << r.chi << " "
                << r.binder << " "
                << r.acceptance << "\n";

            cout << "done: L=" << L
                 << " T=" << T
                 << " binder=" << r.binder
                 << " chi=" << r.chi
                 << "\n";
        }
    }

    out.close();

    double Tc_estimate = estimate_tc_from_binder(results);

    write_gnuplot_script("ising_results.dat", "ising_plots.gp");
    write_readme(warmup_sweeps, measurement_sweeps, J, B, Tc_estimate);

    cout << "\nSimulation finished.\n";
    cout << "Data written to: ising_results.dat\n";
    cout << "Gnuplot script written to: ising_plots.gp\n";
    cout << "Readme written to: Readme.md\n";

    if (Tc_estimate > 0.0) {
        cout << "Estimated Tc from Binder crossing L=20 and L=40: "
             << Tc_estimate << "\n";
    } else {
        cout << "Tc could not be estimated automatically.\n";
    }

    cout << "\nTrying to create PDF plots with gnuplot...\n";

    int ret = system("gnuplot ising_plots.gp");

    if (ret == 0) {
        cout << "Plots written to: ising_plots.pdf\n";
    } else {
        cout << "Could not run gnuplot automatically.\n";
        cout << "Install gnuplot or run manually:\n";
        cout << "  gnuplot ising_plots.gp\n";
    }

    return 0;
}