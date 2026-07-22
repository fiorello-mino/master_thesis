set terminal pngcairo enhanced size 1600,900
set output 'median_mae_models.png'

set xlabel 'Model'
set ylabel 'maxMAE (median) with quartiles'
set grid
set title 'Max MAE median with 25/75 quartiles for different models'

# nomi delle directory/modelli
Model1 = "/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-3_coeffG3e-4_hl3_reload_random/median_mae_error.txt"
Model2 = "/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-4_coeffG3e-4_hl3_reload_random/median_mae_error.txt"
Model3 = "/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-5_coeffG3e-4_hl3_reload_random/median_mae_error.txt"

set xrange [0.5:3.5]

# xtic labels manuali
set xtics ("E=1e-3" 1, "E=1e-4" 2, "E=1e-5" 3)

# stile punti
set style line 1 lc rgb 'blue' pt 7 ps 1.5 lw 2

# Plotto ogni modello come un singolo punto con yerrorbars:
# usando: x : median : q25 : q75
plot \
    Model1 using (1):1:2:3 with yerrorbars ls 1 title "E=1e-3", \
    Model2 using (2):1:2:3 with yerrorbars ls 1 title "E=1e-4", \
    Model3 using (3):1:2:3 with yerrorbars ls 1 title "E=1e-5"

unset output