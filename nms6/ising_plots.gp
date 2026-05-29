set terminal pdfcairo enhanced color size 8,6
set output 'ising_plots.pdf'
set datafile commentschars '#'
set key outside
set grid
set xlabel 'T'
set xzeroaxis

set ylabel '<|m|>'
set title 'Absolute magnetization per spin'
plot \
  'ising_results.dat' using 2:(($1==5)?$6:1/0) with linespoints title 'L=5', \
  'ising_results.dat' using 2:(($1==10)?$6:1/0) with linespoints title 'L=10', \
  'ising_results.dat' using 2:(($1==20)?$6:1/0) with linespoints title 'L=20', \
  'ising_results.dat' using 2:(($1==40)?$6:1/0) with linespoints title 'L=40'

set ylabel '<m^2>'
set title 'Second moment of magnetization'
plot \
  'ising_results.dat' using 2:(($1==5)?$7:1/0) with linespoints title 'L=5', \
  'ising_results.dat' using 2:(($1==10)?$7:1/0) with linespoints title 'L=10', \
  'ising_results.dat' using 2:(($1==20)?$7:1/0) with linespoints title 'L=20', \
  'ising_results.dat' using 2:(($1==40)?$7:1/0) with linespoints title 'L=40'

set ylabel '<m^4>'
set title 'Fourth moment of magnetization'
plot \
  'ising_results.dat' using 2:(($1==5)?$8:1/0) with linespoints title 'L=5', \
  'ising_results.dat' using 2:(($1==10)?$8:1/0) with linespoints title 'L=10', \
  'ising_results.dat' using 2:(($1==20)?$8:1/0) with linespoints title 'L=20', \
  'ising_results.dat' using 2:(($1==40)?$8:1/0) with linespoints title 'L=40'

set ylabel 'chi'
set title 'Magnetic susceptibility per spin'
plot \
  'ising_results.dat' using 2:(($1==5)?$9:1/0) with linespoints title 'L=5', \
  'ising_results.dat' using 2:(($1==10)?$9:1/0) with linespoints title 'L=10', \
  'ising_results.dat' using 2:(($1==20)?$9:1/0) with linespoints title 'L=20', \
  'ising_results.dat' using 2:(($1==40)?$9:1/0) with linespoints title 'L=40'

set ylabel 'U_4'
set title 'Binder cumulant'
plot \
  'ising_results.dat' using 2:(($1==5)?$10:1/0) with linespoints title 'L=5', \
  'ising_results.dat' using 2:(($1==10)?$10:1/0) with linespoints title 'L=10', \
  'ising_results.dat' using 2:(($1==20)?$10:1/0) with linespoints title 'L=20', \
  'ising_results.dat' using 2:(($1==40)?$10:1/0) with linespoints title 'L=40'

set output
