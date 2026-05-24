# plot_amp.gnu

set datafile separator ","

set terminal pngcairo size 1000,600 enhanced font "Helvetica,12"
set output "amplitudes_compare.png"

set xlabel "Time"
set ylabel "Amplitude"
set title "Amplitude vs time (implicit vs explicit)"
set grid
set key top right

plot \
    "amplitude_vs_time.csv"          using 1:2 with linespoints lt 1 pt 7 lw 1.5 title "FEM", \
    "amplitude_vs_time_explicit.csv" using 1:2 with linespoints lt 2 pt 5 lw 1.5 title "Explicit"

unset output