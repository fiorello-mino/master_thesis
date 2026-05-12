# plot_loss.gnu

set term pngcairo size 800,600 enhanced
set output 'compare_train_loss_1.png'

set title 'Train Loss for different models'
set xlabel 'Epoch'
set ylabel 'Loss'
set grid

#set logscale y          # scala logaritmica sull'asse y
#set format y "10^{%L}"  # formato carino per la scala log

set key left top

plot \
     'test_lr_1e-5/train_loss.txt' using ($0+1):1 with lines lw 1.5 lc rgb "#E41A1C" title 'lr = 1e-5 hl = 2', \
     'test_lr_1e-5_1/train_loss.txt' every ::0::424 using ($0+76):1 with lines lw 1.5 lc rgb "#E41A1C" notitle, \
     'test_lr_5e-5/train_loss.txt' using ($0+1):1 with lines lw 1.5 lc rgb "#0072B2" title 'lr = 5e-5 hl = 2', \
     'test_lr_5e-5_hl_3/train_loss.txt' using ($0+1):1 with lines lw 1.5 lc rgb "#4DAF4A" title 'lr = 5e-5 hl = 3'
