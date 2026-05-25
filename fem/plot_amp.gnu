# plot_amp.gnu

set datafile separator ","

set terminal pngcairo size 1200,700 enhanced font "Helvetica,14"
set output "amplitudes_compare.png"

set xlabel "Time"
set ylabel "Amplitude"
set title "Amplitude vs time"
set grid lw 1 lc rgb "#cccccc"
set border lw 1.2

set key top right box opaque spacing 1.2
set tics out

f(x) = 1.0/100.0 * exp(- (2.0*pi)**4 * x * 5e-5)

plot \
    "amplitude_vs_time.csv" using 1:2 with linespoints \
        lc rgb "#1f77b4" pt 7 ps 0.8 lw 2 title "FEM", \
    "amplitude_vs_time_explicit.csv" using 1:2 with linespoints \
        lc rgb "#ff7f0e" pt 5 ps 0.8 lw 2 title "Explicit", \
    f(x) with lines \
        lc rgb "#222222" dt 2 lw 2.5 title "Theory: A(t) = (1/100) e^{-(2* \pi)^4 M_0 t}"

unset output