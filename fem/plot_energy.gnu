# plot_energy.gnu

# set datafile separator ","

set terminal pngcairo size 1200,700 enhanced font "Helvetica,14"
set output "energy_plot_fem.png"

set xlabel "Time"
set ylabel "Surface energy"
set title "Surface energy vs time"
set grid lw 1 lc rgb "#cccccc"
set border lw 1.2

set key top right box opaque spacing 1.2
set tics out

plot \
    "/home/fiorello/mesoEvo/install_seq/pore_8/surf_phys.dat" using 1:3 with linespoints \
        lc rgb "#1f77b4" pt 7 ps 0.2 lw 0.5 notitle

unset output
