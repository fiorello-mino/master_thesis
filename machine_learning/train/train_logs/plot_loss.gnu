# plot_loss.gnu
# Plotta valid_loss.txt e valid_loss.txt in scala semilogy

set term pngcairo size 800,600 enhanced
set output 'valid_loss_50vs100_frames.png'

set title 'Validation Loss 50 frame vs 100 frame lr = 5e-5 (semilog scale)'
set xlabel 'Epoch'
set ylabel 'Loss'
set grid

set logscale y          # scala logaritmica sull'asse y
set format y "10^{%L}"  # formato carino per la scala log

set key left top

# Se le epoche partono da 1, usa l'indice della riga come asse x: 1,2,3,...
plot \
     'test_lr_5e-5/valid_loss.txt' using ($0+1):1 with lines lw 2 lc rgb "#0072B2" title 'Valid loss 50 frame', \
     'test_lr_5e-5_100_frame/valid_loss.txt' using ($0+1):1 with lines lw 2 lc rgb "#E41A1C" title 'Valid loss 100 frame'