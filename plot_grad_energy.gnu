# Gnuplot: confronto tra E_grad (gradiente) e E_tot (totale)
# File: grad_vs_energy_stats.txt
# Colonne:
# 1: time
# 2: median_grad
# 3: p25_grad
# 4: p75_grad
# 5: median_tot
# 6: p25_tot
# 7: p75_tot

set terminal pngcairo size 1400,900 enhanced font "Arial,14"
set output "/data/fiorello/grad_vs_energy/grad_vs_energy_comparison.png"

set title "Energia gradiente vs Energia totale vs tempo" font ",18"
set xlabel "Tempo [s]" font ",14"
set ylabel "Energia" font ",14"

# Griglia
set grid front
set xtics auto
set ytics auto

# Stili con indici numerici
set style line 1 lc rgb "#0060AD" lt 1 lw 2   # mediana gradiente (blu)
set style line 2 lc rgb "#0060AD" lt 2 lw 1   # contorno banda gradiente
set style line 3 lc rgb "#AA0000" lt 1 lw 2   # mediana totale (rosso)
set style line 4 lc rgb "#AA0000" lt 2 lw 1   # contorno banda totale

set style fill solid 0.2 border -1

# Legenda
set key top left font ",14"

plot \
    "/data/fiorello/grad_vs_energy/grad_vs_energy_stats.txt" using 1:2 title "E_grad: mediana" with lines ls 1, \
    "" using 1:3:4 title "E_grad: quartili" with filledcurves lc rgb "#0060AD" fs solid 0.2, \
    "/data/fiorello/grad_vs_energy/grad_vs_energy_stats.txt" using 1:5 title "E_tot: mediana" with lines ls 3, \
    "" using 1:6:7 title "E_tot: quartili" with filledcurves lc rgb "#AA0000" fs solid 0.2