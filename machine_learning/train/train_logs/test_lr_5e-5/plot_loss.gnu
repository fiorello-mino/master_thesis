# plot_loss.gnu
# Plotta train_loss.txt e valid_loss.txt in scala semilogy

set term pngcairo size 800,600 enhanced
set output 'loss_curves.png'

set title 'Training vs Validation Loss'
set xlabel 'Epoch'
set ylabel 'Loss'
set grid

set logscale y          # scala logaritmica sull'asse y
set format y "10^{%L}"  # formato carino per la scala log

set key left top

# Se le epoche partono da 1, usa l'indice della riga come asse x: 1,2,3,...
plot \
     'train_loss.txt' using ($0+1):1 with lines lw 2 lc rgb '#1f77b4' title 'Train loss', \
     'valid_loss.txt' using ($0+1):1 with lines lw 2 lc rgb '#ff7f0e' title 'Valid loss'