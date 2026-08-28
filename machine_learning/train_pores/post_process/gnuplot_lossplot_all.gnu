# gnuplot_lossplot_per_model.gnu

BaseDir = "/home/fiorello/master_thesis/machine_learning/train_pores/train_logs"
Models  = "coeffE1e-3_hl3 coeffE1e-3_hl3_reload_random coeffE1e-3_coeffG3e-4_hl3_reload_random"

set terminal pngcairo size 1200,800 enhanced font ',12'

set logscale y
set format y "10^{%L}"    # etichette come 10^-1, 10^-2, ...

set xlabel 'Epoch (index)'
set ylabel 'Loss'
set grid

set key top right box opaque

set style line 1 lc rgb '#1f77b4' lw 2   # train (blu)
set style line 2 lc rgb '#ff7f0e' lw 2   # valid (arancione)

do for [i=1:words(Models)] {
    ModelName = word(Models,i)
    set title sprintf('Training and validation loss (semilogy) - %s', ModelName)

    set output sprintf('loss_%s.png', ModelName)

    plot \
        BaseDir."/".ModelName."/train_loss.txt" using 0:1 with lines ls 1 title "train", \
        BaseDir."/".ModelName."/valid_loss.txt" using 0:1 with lines ls 2 title "valid"

    unset output
}