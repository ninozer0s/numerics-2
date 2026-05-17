set terminal pngcairo size 900,600
set output 'potential_and_psi0.png'
set xlabel 'r / bohr'
set xrange [7:10]
set title 'Ground-state wavefunction around potential minimum'
set ylabel 'V(r) / Hartree'
set y2label 'scaled |psi_0(r)|'
set y2tics
set y2range [0:1.05]
plot 'pot.dat' using 1:2 with lines axis x1y1 title 'V(r)', \
     'psi0_scaled.dat' using 1:2 with lines axis x1y2 title 'scaled |psi_0|'
