# plot_loss.gnu

set term pngcairo size 800,600 enhanced
set output 'MSE_test_64_div_vs_cons.png'

set title 'Comparing overall MSE external test for models trained with divergence vs conservative (log scale)'
set xlabel 'Test set'
set ylabel 'MSE'
set grid

set logscale y
#set format y "10^{%L}"

set key left top

plot \
     'dataset_64_2_ext_test/lr_5e-5_hl_3_2_tr_10/errors.txt' using ($0):5 with lines lw 1.5 lc rgb "#0072B2" title 'divergence', \
     'dataset_64_2_ext_test/lr_5e-5_hl_3_cons_2_tr_10/errors.txt' using ($0):5 with lines lw 1.5 lc rgb "#E41A1C" title 'conservative'
     #'ext_test_lr_5e-5_hl_3_2/errors.txt' using ($0):5 with lines lw 1.5 lc rgb "#E41A1C" title 'train from frame 3, test from frame 1'
     #'ext_test_lr_5e-5_100_frame/dataset_64_random/errors.txt' using ($0):2 with lines lw 1.5 lc rgb "#4DAF4A" title 'lr = 5e-5 100 frames', \
     #'ext_test_lr_5e-5_hl_3/dataset_64_random/errors_validation_set.txt' every ::0::99 using ($0):5 with lines lw 1.5 lc rgb "#E41A1C" title 'valid set'

# rgb "#984EA3" viola
# rgb "#E41A1C" rosso
# rgb "#0072B2" blu
# rgb "#4DAF4A" verde
