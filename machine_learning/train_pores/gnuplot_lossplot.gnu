set terminal pngcairo size 1200,800 enhanced font ',12'
set output 'train_valid_loss_all_models.png'

set logscale y
set xlabel 'Epoch'
set ylabel 'Loss'
set grid
set title 'Training and validation loss (semilogy) for all models'

# directory base
BaseDir = "/home/fiorello/master_thesis/machine_learning/train_pores/train_logs"

# lista dei modelli (cartelle) da plottare
Models = "coeffE1e-3_hl3 coeffE1e-3_hl3_reload_random coeffE1e-3_coeffG3e-4_hl3_reload_random coeffE1e-3_coeffG3e-4_hl3_reload_random_bin"

# stile colori, uno per modello
set style line 1 lc rgb '#1f77b4' lw 2
set style line 2 lc rgb '#ff7f0e' lw 2
set style line 3 lc rgb '#2ca02c' lw 2
set style line 4 lc rgb '#d62728' lw 2

# train = linea piena, valid = tratteggiata
TrainLT = 1         # linetype 1 (solid)
ValidLT = 2         # linetype 2 (dashed)

# legenda chiara: "model_A train", "model_A valid", ecc.
set key outside right top vertical Left reverse

plot \
    for [i=1:words(Models)] \
        BaseDir."/".word(Models,i)."/train_loss.txt" using 1:2 \
            with lines ls i lt TrainLT title word(Models,i)." train", \
    for [i=1:words(Models)] \
        BaseDir."/".word(Models,i)."/valid_loss.txt" using 1:2 \
            with lines ls i lt ValidLT title word(Models,i)." valid"

unset output