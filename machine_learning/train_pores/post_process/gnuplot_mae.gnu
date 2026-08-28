set terminal pngcairo size 900,700 enhanced font ',12'
set output 'median_mae_models.png'

# stile generale pulito
unset key
set border lw 1.5
set grid ytics lc rgb "#dddddd" lw 1
set tics out nomirror

set xlabel 'Model'
set ylabel 'maxMAE (median) with quartiles'
set title 'Max MAE median with 25/75 quartiles for different models'

Model1 = "/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-3_hl3/median_mae.txt"
Model2 = "/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-3_hl3_reload_random/median_mae.txt"
Model3 = "/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-3_coeffG3e-4_hl3_reload_random/median_mae.txt"
Model4 = "/data/fiorello/pores/ext_test/ext_test_var_depth/coeffE1e-3_coeffG3e-4_hl3_reload_random/median_mae_bin.txt"

set xrange [0.5:4.5]
set xtics ("no reload" 1, "reload" 2, "reload+grad" 3, "reload+grad+pp" 4)

# box stretti
set boxwidth 0.12 relative
set style fill solid 0.25 border

# colori di prima
set style line 1 lc rgb '#1f77b4' lw 1.5   # blu
set style line 2 lc rgb '#ff7f0e' lw 1.5   # arancione
set style line 3 lc rgb '#2ca02c' lw 1.5   # verde
set style line 4 lc rgb '#d62728' lw 1.5   # rosso

# mediane un po' più spesse
set style line 11 lc rgb '#1f77b4' lw 2.5
set style line 12 lc rgb '#ff7f0e' lw 2.5
set style line 13 lc rgb '#2ca02c' lw 2.5
set style line 14 lc rgb '#d62728' lw 2.5

# file: col1=median, col2=q25, col3=q75
plot \
    Model1 using (1):2:2:3:3 with candlesticks whiskerbars ls 1, \
    ''     using (1):1:1:1:1 with candlesticks ls 11, \
    Model2 using (2):2:2:3:3 with candlesticks whiskerbars ls 2, \
    ''     using (2):1:1:1:1 with candlesticks ls 12, \
    Model3 using (3):2:2:3:3 with candlesticks whiskerbars ls 3, \
    ''     using (3):1:1:1:1 with candlesticks ls 13, \
    Model4 using (4):2:2:3:3 with candlesticks whiskerbars ls 4, \
    ''     using (4):1:1:1:1 with candlesticks ls 14

unset output
