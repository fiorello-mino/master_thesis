### plot_compare.gp ###################################################
# Cambia solo qui i path e i label
# -------------------------------------------------------------------
#errorsA = 'ext_test_64_2/lr5e-5_hl3_2_tr10/errors.txt'
#errorsB = 'ext_test_64_2/lr1e-4_b4_k7_hl2_ch16_seq20_ramp5_wd2e-5/errors.txt'
#medianA = 'ext_test_64_2/lr5e-5_hl3_2_tr10/median_energy_error.txt'
#medianB = = 'ext_test_64_2/lr1e-4_b4_k7_hl2_ch16_seq20_ramp5_wd2e-5/median_energy_error.txt'
#evoA    = 'ext_test_64_2/lr1e-4_b8_k3_hl2_ch16_seq20/0000/evo.txt'
#evoB    = 'ext_test_64_2/lr1e-4_b4_k7_hl2_ch16_seq20_ramp5_wd2e-5/0000/evo.txt'

#labelA  = 'lr5e-5_k5_hl3'
#labelB  = 'lr1e-4_k7_hl2'

#outdir  = 'plots/prec_vs_new_2/'
# -------------------------------------------------------------------

set terminal pngcairo size 900,700 enhanced font ',12'
set datafile separator whitespace
set key top left

# colonne errors.txt: 1:id  2:maxMAE  3:maxMSE  4:overallMAE  5:overallMSE
# colonne evo.txt:    1:MAE  2:MSE  3:avg_True  4:avg_Pred
#                     5:min_True  6:min_Pred  7:max_True  8:max_Pred
#                     9:E_True  10:E_Pred
# asse x nei plot evo: $0 = indice di riga = frame index

#######################################################################
# 1) overall MAE per sequenza
#######################################################################
set output outdir.'overallMAE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Overall MAE'
set grid
plot errorsA using 1:4 with lines lc rgb 'blue' lw 2 title labelA, \
     errorsB using 1:4 with lines lc rgb 'red'  lw 2 title labelB

#######################################################################
# 2) overall MSE per sequenza
#######################################################################
set output outdir.'overallMSE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Overall MSE'
set grid
plot errorsA using 1:5 with lines lc rgb 'blue' lw 2 title labelA, \
     errorsB using 1:5 with lines lc rgb 'red'  lw 2 title labelB

#######################################################################
# 3) max MAE per sequenza
#######################################################################
set output outdir.'maxMAE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Max MAE'
set grid
plot errorsA using 1:2 with lines lc rgb 'blue' lw 2 title labelA, \
     errorsB using 1:2 with lines lc rgb 'red'  lw 2 title labelB

#######################################################################
# 4) max MSE per sequenza
#######################################################################
set output outdir.'maxMSE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Max MSE'
set grid
plot errorsA using 1:3 with lines lc rgb 'blue' lw 2 title labelA, \
     errorsB using 1:3 with lines lc rgb 'red'  lw 2 title labelB

#######################################################################
# 5) mediana dell'errore relativo per dt
#######################################################################
set output outdir.'median_rel_error_compare.png'
set xlabel "time"
set ylabel "median |ΔE| / |E_true|"
set grid

set style fill transparent solid 0.2 noborder

plot \
    medianA using 1:3:4 with filledcurves lc rgb 'blue'  title labelA.'_IQR', \
    medianB using 1:3:4 with filledcurves lc rgb 'red'   title labelB.'_IQR', \
    medianA using 1:2      with lines       lc rgb 'blue' lw 2 title labelA, \
    medianB using 1:2      with lines       lc rgb 'red'  lw 2 title labelB











# #######################################################################
# # 5) MAE(t) seq 0000
# #######################################################################
# set output outdir.'seq0000_MAE_vs_t.png'
# set xlabel 'Time step'
# set ylabel 'MAE'
# set grid
# plot evoA using ($0):1 with lines lc rgb 'blue' lw 2 title labelA, \
#      evoB using ($0):1 with lines lc rgb 'red'  lw 2 title labelB

# #######################################################################
# # 6) MSE(t) seq 0000
# #######################################################################
# set output outdir.'seq0000_MSE_vs_t.png'
# set xlabel 'Time step'
# set ylabel 'MSE'
# set grid
# plot evoA using ($0):2 with lines lc rgb 'blue' lw 2 title labelA, \
#      evoB using ($0):2 with lines lc rgb 'red'  lw 2 title labelB

# #######################################################################
# # 7) avg_true / avg_pred — modello A
# #######################################################################
# set output outdir.'seq0000_avg.png'
# set xlabel 'Time step'
# set ylabel 'Average field'
# set grid
# plot evoA using ($0):3 with lines lc rgb 'black' lw 2 title 'avg\_true '.labelA, \
#      evoA using ($0):4 with lines lc rgb 'blue'  lw 2 title 'avg\_pred '.labelA, \
#      evoB using ($0):4 with lines lc rgb 'red'   lw 2 title 'avg\_pred '.labelB

# #######################################################################
# # 8) energia nel tempo — E_true comune + E_pred A e B
# #######################################################################
# set output outdir.'seq0000_energy.png'
# set xlabel 'Time step'
# set ylabel 'Energy'
# set grid
# plot evoA using ($0):9  with lines lc rgb 'black' lw 2 title 'E\_true', \
#      evoA using ($0):10 with lines lc rgb 'blue'  lw 2 title 'E\_pred '.labelA, \
#      evoB using ($0):10 with lines lc rgb 'red'   lw 2 title 'E\_pred '.labelB

# #######################################################################
# # 9) errore relativo sull'energia
# #######################################################################
# set output outdir.'seq0000_rel_err_E.png'
# set xlabel 'Time step'
# set ylabel 'Relative error on E'
# set grid
# plot evoA using ($0):(abs($10-$9)/abs($9)) with lines lc rgb 'blue' lw 2 title labelA, \
#      evoB using ($0):(abs($10-$9)/abs($9)) with lines lc rgb 'red'  lw 2 title labelB

# #######################################################################
# # 10) max_true / max_pred
# #######################################################################
# set output outdir.'seq0000_max.png'
# set xlabel 'Time step'
# set ylabel 'Max field value'
# set grid
# plot evoA using ($0):7 with lines lc rgb 'black' lw 2 title 'max\_true '.labelA, \
#      evoA using ($0):8 with lines lc rgb 'blue'  lw 2 title 'max\_pred '.labelA, \
#      evoB using ($0):8 with lines lc rgb 'red'   lw 2 title 'max\_pred '.labelB

unset output
### fine script ########################################################