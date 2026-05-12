# plot_loss.gnu
# Plotta valid_loss.txt e valid_loss.txt in scala semilogy

set term pngcairo size 800,600 enhanced
set output 'MSE_64_random.png'

set title 'Overall MSE for different models, dataset 64x64 random'
set xlabel 'Test set'
set ylabel 'MSE'
set grid

#set logscale y
#set format y "10^{%L}"

set key left top

plot \
     'ext_test_lr_1e-4/errors.txt' using ($0):5 with lines lw 2 lc rgb "#0072B2" title 'lr = 1e-4', \
     'ext_test_lr_5e-5/dataset_64_random/errors.txt' using ($0):5 with lines lw 2 lc rgb "#E41A1C" title 'lr = 5e-5', \
     'ext_test_lr_5e-5_100_frame/dataset_64_random/errors.txt' using ($0):5 with lines lw 2 lc rgb "#4DAF4A" title 'lr = 5e-5 100 frames', \
     'ext_test_lr_5e-5_hl_3/dataset_64_random/errors.txt' using ($0):5 with lines lw 2 lc rgb "#984EA3" title 'lr = 5e-5 hl = 3'
