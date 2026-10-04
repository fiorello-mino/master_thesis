### compare_models.gnu ################################################
# Confronto tra due modelli A e B
#
# Variabili ricevute da run_plot.sh:
#
#   errorsA
#   errorsB
#   medianA
#   medianB
#   labelA
#   labelB
#   outdir
#
#######################################################################


# =====================================================================
# GENERAL STYLE
# =====================================================================

set terminal pngcairo size 900,700 enhanced font ',12'

set datafile separator whitespace

set key top left
set grid



# =====================================================================
# FILE STRUCTURE
# =====================================================================

# errors.txt
#
#  1 : id
#  2 : maxMAE
#  3 : maxMSE
#  4 : overallMAE
#  5 : overallMSE
#  6 : max(symDiff)
#  7 : avg(symDiff)
#  8 : phi_min_sequence
#  9 : phi_max_sequence
# 10 : domains_final_true
# 11 : domains_final_pred
# 12 : delta_domains
#
# La colonna 1 e' una stringa.
# Per Sequence id usiamo:
#
#     ($0+1)
#


# medians.txt
#
#  1 : time
#
#  2 : median_energy_error
#  3 : p25_energy_error
#  4 : p75_energy_error
#
#  5 : median_mass_error
#  6 : p25_mass_error
#  7 : p75_mass_error
#
#  8 : median_phi_min
#  9 : p25_phi_min
# 10 : p75_phi_min
#
# 11 : median_phi_max
# 12 : p25_phi_max
# 13 : p75_phi_max



#######################################################################
# 1) OVERALL MAE PER SEQUENZA
#######################################################################

set output outdir.'overallMAE_compare.png'

set xlabel 'Sequence id'
set ylabel 'Overall MAE'

set xrange [1:50]
set xtics 5

plot \
    errorsA using ($0+1):4 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.6 \
        title labelA, \
    errorsB using ($0+1):4 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.6 \
        title labelB



#######################################################################
# 2) OVERALL MSE PER SEQUENZA
#######################################################################

set output outdir.'overallMSE_compare.png'

set xlabel 'Sequence id'
set ylabel 'Overall MSE'

plot \
    errorsA using ($0+1):5 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.6 \
        title labelA, \
    errorsB using ($0+1):5 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.6 \
        title labelB



#######################################################################
# 3) MAX MAE PER SEQUENZA
#######################################################################

set output outdir.'maxMAE_compare.png'

set xlabel 'Sequence id'
set ylabel 'Max MAE'

plot \
    errorsA using ($0+1):2 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.6 \
        title labelA, \
    errorsB using ($0+1):2 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.6 \
        title labelB



#######################################################################
# 4) MAX MSE PER SEQUENZA
#######################################################################

set output outdir.'maxMSE_compare.png'

set xlabel 'Sequence id'
set ylabel 'Max MSE'

plot \
    errorsA using ($0+1):3 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.6 \
        title labelA, \
    errorsB using ($0+1):3 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.6 \
        title labelB



#######################################################################
# 5) MAX SYMMETRIC DIFFERENCE PER SEQUENZA
#######################################################################

set output outdir.'max_symDiff_compare.png'

set xlabel 'Sequence id'
set ylabel 'Max symmetric difference'

plot \
    errorsA using ($0+1):6 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.6 \
        title labelA, \
    errorsB using ($0+1):6 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.6 \
        title labelB



#######################################################################
# 6) AVERAGE SYMMETRIC DIFFERENCE PER SEQUENZA
#######################################################################

set output outdir.'avg_symDiff_compare.png'

set xlabel 'Sequence id'
set ylabel 'Average symmetric difference'

plot \
    errorsA using ($0+1):7 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.6 \
        title labelA, \
    errorsB using ($0+1):7 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.6 \
        title labelB



# =====================================================================
# TEMPORAL PLOTS
# =====================================================================

unset xrange
set xtics auto

set style fill transparent solid 0.20 noborder



#######################################################################
# 7) RELATIVE ENERGY ERROR:
#    MEDIAN + P25/P75
#######################################################################

set output outdir.'median_energy_error_compare.png'

set xlabel 'Time'
set ylabel 'Relative energy error'

plot \
    medianA using 1:3:4 with filledcurves \
        lc rgb 'blue' \
        title labelA.'_IQR', \
    medianB using 1:3:4 with filledcurves \
        lc rgb 'red' \
        title labelB.'_IQR', \
    medianA using 1:2 with lines \
        lc rgb 'blue' lw 2 \
        title labelA, \
    medianB using 1:2 with lines \
        lc rgb 'red' lw 2 \
        title labelB



#######################################################################
# 8) RELATIVE MASS ERROR:
#    MEDIAN + P25/P75
#######################################################################

set output outdir.'median_mass_error_compare.png'

set xlabel 'Time'
set ylabel 'Relative mass error'

plot \
    medianA using 1:6:7 with filledcurves \
        lc rgb 'blue' \
        title labelA.'_IQR', \
    medianB using 1:6:7 with filledcurves \
        lc rgb 'red' \
        title labelB.'_IQR', \
    medianA using 1:5 with lines \
        lc rgb 'blue' lw 2 \
        title labelA, \
    medianB using 1:5 with lines \
        lc rgb 'red' lw 2 \
        title labelB



#######################################################################
# 9) MIN(PHI) VS TIME
#
# Per ogni simulazione s e timestep t:
#
#     phi_min_s(t) = min_xyz phi_s(t)
#
# medians.txt contiene poi:
#
#     median_s(phi_min_s(t))
#     p25_s(phi_min_s(t))
#     p75_s(phi_min_s(t))
#######################################################################

set output outdir.'phi_min_vs_time_compare.png'

set xlabel 'Time'
set ylabel 'min(phi)'

plot \
    medianA using 1:9:10 with filledcurves \
        lc rgb 'blue' \
        title labelA.'_IQR', \
    medianB using 1:9:10 with filledcurves \
        lc rgb 'red' \
        title labelB.'_IQR', \
    medianA using 1:8 with lines \
        lc rgb 'blue' lw 2 \
        title labelA, \
    medianB using 1:8 with lines \
        lc rgb 'red' lw 2 \
        title labelB



#######################################################################
# 10) MAX(PHI) VS TIME
#
# Per ogni simulazione s e timestep t:
#
#     phi_max_s(t) = max_xyz phi_s(t)
#
# medians.txt contiene poi:
#
#     median_s(phi_max_s(t))
#     p25_s(phi_max_s(t))
#     p75_s(phi_max_s(t))
#######################################################################

set output outdir.'phi_max_vs_time_compare.png'

set xlabel 'Time'
set ylabel 'max(phi)'

plot \
    medianA using 1:12:13 with filledcurves \
        lc rgb 'blue' \
        title labelA.'_IQR', \
    medianB using 1:12:13 with filledcurves \
        lc rgb 'red' \
        title labelB.'_IQR', \
    medianA using 1:11 with lines \
        lc rgb 'blue' lw 2 \
        title labelA, \
    medianB using 1:11 with lines \
        lc rgb 'red' lw 2 \
        title labelB



# =====================================================================
# PER-SEQUENCE PLOTS
# =====================================================================

set xrange [1:50]
set xtics 5



#######################################################################
# 11) MINIMO GLOBALE DELL'INTERA SEQUENZA
#
# Per ogni simulazione:
#
#     min_{t,x,y,z} phi_pred
#
# errors.txt colonna 8
#######################################################################

set output outdir.'global_phi_min_per_sequence_compare.png'

set xlabel 'Sequence id'
set ylabel 'Global sequence min(phi)'

plot \
    errorsA using ($0+1):8 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.7 \
        title labelA, \
    errorsB using ($0+1):8 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.7 \
        title labelB



#######################################################################
# 12) MASSIMO GLOBALE DELL'INTERA SEQUENZA
#
# Per ogni simulazione:
#
#     max_{t,x,y,z} phi_pred
#
# errors.txt colonna 9
#######################################################################

set output outdir.'global_phi_max_per_sequence_compare.png'

set xlabel 'Sequence id'
set ylabel 'Global sequence max(phi)'

plot \
    errorsA using ($0+1):9 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.7 \
        title labelA, \
    errorsB using ($0+1):9 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.7 \
        title labelB



#######################################################################
# 13) DOMAINS TRUE AL FRAME FINALE
#
# errors.txt colonna 10
#######################################################################

set output outdir.'final_domains_true_compare.png'

set xlabel 'Sequence id'
set ylabel 'Final connected domains - true'

set ytics 1

plot \
    errorsA using ($0+1):10 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.8 \
        title labelA, \
    errorsB using ($0+1):10 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.8 \
        title labelB



#######################################################################
# 14) DOMAINS PRED AL FRAME FINALE
#
# errors.txt colonna 11
#######################################################################

set output outdir.'final_domains_pred_compare.png'

set xlabel 'Sequence id'
set ylabel 'Final connected domains - predicted'

set ytics 1

plot \
    errorsA using ($0+1):11 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.8 \
        title labelA, \
    errorsB using ($0+1):11 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.8 \
        title labelB



#######################################################################
# 15) DELTA NUMERO DI DOMINI
#
# errors.txt:
#
#     colonna 12 = N_pred - N_true
#
# Delta N = 0:
#     numero corretto di domini
#
# Delta N > 0:
#     troppi domini predetti
#
# Delta N < 0:
#     troppo pochi domini predetti
#######################################################################

set output outdir.'final_domains_delta_compare.png'

set xlabel 'Sequence id'
set ylabel 'Delta N = N_pred - N_true'

set ytics 1


# linea Delta N = 0

set arrow 1 \
    from graph 0, first 0 \
    to graph 1, first 0 \
    nohead \
    dt 2 \
    lw 1.5 \
    lc rgb 'black' \
    back


plot \
    errorsA using ($0+1):12 with linespoints \
        lc rgb 'blue' lw 2 pt 7 ps 0.8 \
        title labelA, \
    errorsB using ($0+1):12 with linespoints \
        lc rgb 'red' lw 2 pt 7 ps 0.8 \
        title labelB


unset arrow 1



# =====================================================================
# RESET
# =====================================================================

unset xrange

set xtics auto
set ytics auto

unset output


### fine script #######################################################
