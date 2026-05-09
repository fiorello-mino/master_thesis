reset

# -----------------------------
# 1) Costruzione file temporanei
# -----------------------------
system("head -n 75 test_lr_1e-5/valid_loss.txt > tmp_lr_1e-5.dat")
system("head -n 425 test_lr_1e-5_1/valid_loss.txt >> tmp_lr_1e-5.dat")
system("head -n 500 test_lr_5e-5/valid_loss.txt > tmp_lr_5e-5.dat")

# -----------------------------
# 2) Trova minimo ed epoca con awk
# -----------------------------
min_lr1   = real(system("awk 'NR==1{min=$1; ep=1} $1<min{min=$1; ep=NR} END{print min}' tmp_lr_1e-5.dat"))
epoch_lr1 = real(system("awk 'NR==1{min=$1; ep=1} $1<min{min=$1; ep=NR} END{print ep}' tmp_lr_1e-5.dat"))

min_lr2   = real(system("awk 'NR==1{min=$1; ep=1} $1<min{min=$1; ep=NR} END{print min}' tmp_lr_5e-5.dat"))
epoch_lr2 = real(system("awk 'NR==1{min=$1; ep=1} $1<min{min=$1; ep=NR} END{print ep}' tmp_lr_5e-5.dat"))

# -----------------------------
# 3) Plot
# -----------------------------
set terminal pngcairo size 1000,650 enhanced
set output 'compare_valid_loss.png'

set title 'Validation loss: lr = 1e-5 vs lr = 5e-5'
set xlabel 'Epoch'
set ylabel 'Validation loss'
set grid
set logscale y
set format y "10^{%L}"
set key top right

# Palette scientifica classica
color1 = "#0072B2"   # blu scuro
color2 = "#D55E00"   # vermiglio

# Label solo con numero epoca
set label 1 sprintf("min lr=1e-5\n(epoch=%d)", int(epoch_lr1)) \
   at epoch_lr1, min_lr1 offset -3.0,-1.5 tc rgb color1

set label 2 sprintf("min lr=5e-5\n(epoch=%d)", int(epoch_lr2)) \
   at epoch_lr2, min_lr2 offset -4.0,-1.5 tc rgb color2

plot \
    'tmp_lr_1e-5.dat' using ($0+1):1 with lines lw 2.5 lc rgb color1 title 'lr = 1e-5', \
    'tmp_lr_5e-5.dat' using ($0+1):1 with lines lw 2.5 lc rgb color2 title 'lr = 5e-5', \
    '+' using (epoch_lr1):(min_lr1) with points pt 7 ps 1.8 lc rgb color1 notitle, \
    '+' using (epoch_lr2):(min_lr2) with points pt 7 ps 1.8 lc rgb color2 notitle'

unset output