### plot_compare.gp ###################################################
# Cambia solo qui i path e i label
# -------------------------------------------------------------------
errorsA = 'ext_test_64_2/lr1e-4_b8_k3_hl2_ch16_seq20/errors.txt'
errorsB = 'ext_test_64_2/lr1e-4_b8_k3_hl2_ch16_seq30/errors.txt'
evoA    = 'ext_test_64_2/lr1e-4_b8_k3_hl2_ch16_seq20/0000/evo.txt'
evoB    = 'ext_test_64_2/lr1e-4_b8_k3_hl2_ch16_seq30/0000/evo.txt'

labelA  = 'subseq\_max = 20'
labelB  = 'subseq\_max = 30'

outdir  = 'plots/seq20_vs_seq30/'
# -------------------------------------------------------------------

set terminal pngcairo size 900,700 enhanced font ',12'
set datafile separator whitespace
set key top left

# colonne errors.txt: 1:id  2:maxMAE  3:maxMSE  4:overallMAE  5:overallMSE
# colonne evo.txt:    1:t   2:MAE     3:MSE      4:avg_True    5:avg_Pred
#                     6:min_True 7:min_Pred 8:max_True 9:max_Pred
#                     10:E_True  11:E_Pred

#######################################################################
# 1) overall MAE per sequenza
#######################################################################
set output outdir.'overallMAE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Overall MAE'
set grid
plot errorsA using 1:4 with points pt 7 ps 1.2 lc rgb 'blue' title labelA, \
     errorsB using 1:4 with points pt 5 ps 1.2 lc rgb 'red'  title labelB

#######################################################################
# 2) overall MSE per sequenza
#######################################################################
set output outdir.'overallMSE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Overall MSE'
set grid
plot errorsA using 1:5 with points pt 7 ps 1.2 lc rgb 'blue' title labelA, \
     errorsB using 1:5 with points pt 5 ps 1.2 lc rgb 'red'  title labelB

#######################################################################
# 3) max MAE per sequenza
#######################################################################
set output outdir.'maxMAE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Max MAE'
set grid
plot errorsA using 1:2 with impulses lc rgb 'blue' title labelA, \
     errorsB using 1:2 with impulses lc rgb 'red'  title labelB

#######################################################################
# 4) max MSE per sequenza
#######################################################################
set output outdir.'maxMSE_compare.png'
set xlabel 'Sequence id'
set ylabel 'Max MSE'
set grid
plot errorsA using 1:3 with impulses lc rgb 'blue' title labelA, \
     errorsB using 1:3 with impulses lc rgb 'red'  title labelB

#######################################################################
# 5) MAE(t) seq 0000
#######################################################################
set output outdir.'seq0000_MAE_vs_t.png'
set xlabel 'Time step'
set ylabel 'MAE'
set grid
plot evoA using 1:2 with lines lc rgb 'blue' lw 2 title labelA, \
     evoB using 1:2 with lines lc rgb 'red'  lw 2 title labelB

#######################################################################
# 6) MSE(t) seq 0000
#######################################################################
set output outdir.'seq0000_MSE_vs_t.png'
set xlabel 'Time step'
set ylabel 'MSE'
set grid
plot evoA using 1:3 with lines lc rgb 'blue' lw 2 title labelA, \
     evoB using 1:3 with lines lc rgb 'red'  lw 2 title labelB

#######################################################################
# 7) avg_true / avg_pred — modello A
#######################################################################
set output outdir.'seq0000_avg.png'
set xlabel 'Time step'
set ylabel 'Average field'
set grid
plot evoA using 1:4 with lines lc rgb 'black' lw 2 title 'avg\_true', \
     evoA using 1:5 with lines lc rgb 'blue'  lw 2 title 'avg\_pred '.labelA, \
     evoB using 1:5 with lines lc rgb 'red'   lw 2 title 'avg\_pred '.labelB

#######################################################################
# 8) energia nel tempo — E_true comune + E_pred A e B
#######################################################################
set output outdir.'seq0000_energy.png'
set xlabel 'Time step'
set ylabel 'Energy'
set grid
plot evoA using 1:10 with lines lc rgb 'black' lw 2 title 'E\_true', \
     evoA using 1:11 with lines lc rgb 'blue'  lw 2 title 'E\_pred '.labelA, \
     evoB using 1:11 with lines lc rgb 'red'   lw 2 title 'E\_pred '.labelB

#######################################################################
# 9) errore relativo sull'energia
#######################################################################
set output outdir.'seq0000_rel_err_E.png'
set xlabel 'Time step'
set ylabel 'Relative error on E'
set grid
plot evoA using 1:(($10>0 && $10==$10 && $11==$11) ? abs($11-$10)/abs($10) : 1/0) with lines lc rgb 'blue' lw 2 title 'labelA', \
     evoB using 1:(($10>0 && $10==$10 && $11==$11) ? abs($11-$10)/abs($10) : 1/0) with lines lc rgb 'red' lw 2 title 'labelB'

#######################################################################
# 10) max_true / max_pred
#######################################################################
set output outdir.'seq0000_max.png'
set xlabel 'Time step'
set ylabel 'Max field value'
set grid
plot evoA using 1:8 with lines lc rgb 'black' lw 2 title 'max\_true', \
     evoA using 1:9 with lines lc rgb 'blue'  lw 2 title 'max\_pred '.labelA, \
     evoB using 1:9 with lines lc rgb 'red'   lw 2 title 'max\_pred '.labelB


unset output
### fine script ########################################################