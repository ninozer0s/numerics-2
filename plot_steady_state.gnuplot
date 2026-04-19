set terminal pdfcairo
set output 'steady_state.pdf'
set xlabel 'site index'
set ylabel 'steady state photons'
set title 'Steady state of scattering model'
plot 'steady_state.dat' using 1:2 with lines lw 2 title 'Reference', \
     'steady_state.dat' u 1:3 w lines dt 2 title 'MINRES (r=20)'
