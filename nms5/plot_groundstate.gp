set terminal pngcairo size 900,600
set output 'psi0.png'
set xlabel 'r / bohr'
set ylabel 'psi_0(r)'
set xrange [7:10]
set title 'Ground-state wavefunction around potential minimum'
plot 'psi0.dat' using 1:2 with lines title 'psi_0'
