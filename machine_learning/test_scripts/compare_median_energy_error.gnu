# compare_median_energy_error.gnu

reset

labelA = 'lr5e-5_k5_hl3'
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
fileA = 'ext_test_64_2/lr5e-5_hl3_2_tr10/median_energy_error.txt'
fileB = 'ext_test_64_2/lr1e-4_b4_k7_hl2_ch16_seq20_ramp5_wd2e-5/median_energy_error.txt'

plot \
    fileA using 1:2 with lines lc rgb 'blue' lw 2 title labelA, \
    fileB using 1:2 with lines lc rgb 'red'  lw 2 title labelB