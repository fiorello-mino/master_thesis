# compare_median_energy_error.gnu

reset

labelA = 'lr5e-5_hl3'
labelB = 'lr1e-4_k7_hl2'

outdir  = 'plots/prec_vs_new_2/'
set terminal pngcairo size 1200,800 enhanced font 'Arial,18'
set output outdir.'median_energy_comparison.png'

set title "Median relative energy error"
set xlabel "time"
set ylabel "median |ΔE| / |E_true|"
set grid

set key top right

# file
fileA = "median_run1.txt"
fileB = "median_run2.txt"

plot \
    fileA using 1:2 with lines lc rgb 'blue' lw 2 title labelA, \
    fileB using 1:2 with lines lc rgb 'red'  lw 2 title labelB