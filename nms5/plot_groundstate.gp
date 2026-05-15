set terminal pngcairo size 900,600
set output 'psi0.png'
set xlabel 'r / bohr'
set ylabel 'normalized psi_0(r)'
set xrange [7:10]
plot 'psi0.dat' using 1:2 with lines title 'psi_0'
