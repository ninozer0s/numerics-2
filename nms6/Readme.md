# Monte-Carlo-Simulation des 2D-Ising-Modells

## Methode

Simuliert wird das Ising-Modell auf einem quadratischen Gitter mit periodischen Randbedingungen.
Verwendet wird der Single-Spin-Flip-Metropolis-Algorithmus.

Hamiltonian:

`H = -J sum_<ij> s_i s_j - B sum_i s_i`

Ein Sweep besteht aus `L*L` versuchten Spin-Flips.

## Parameter

- J = 1
- B = 0
- Systemgroessen: L = 5, 10, 20, 40
- Warm-up sweeps: 5000
- Measurement sweeps: 20000
- Temperaturen: T = 1 bis 7, plus feines Raster um T = 2.269
- Zufallszahlengenerator: std::mt19937

## Gemessene Groessen

Die Magnetisierung wird pro Spin verwendet: m = M / (L*L).

- <|m|>
- <m^2>
- <m^4>
- Suszeptibilitaet: chi = N * (<m^2> - <|m|>^2) / T
- Binder-Kumulante: U4 = 1 - <m^4> / (3 <m^2>^2)

## Kritische Temperatur

Aus dem Schnittpunkt der Binder-Kumulanten fuer L=20 und L=40 ergibt sich grob:

`Tc approx 2.26243`

Der erwartete Wert fuer das 2D-Ising-Modell auf dem Quadratgitter bei J=1 und kB=1 ist ungefaehr:

`Tc approx 2.269`

## Output

- ising_results.dat: Messdaten
- ising_plots.pdf: Plots aller geforderten Groessen
- ising_plots.gp: automatisch erzeugtes gnuplot-Skript
