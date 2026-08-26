import numpy as np
import matplotlib.pyplot as plt

# Carica il file allungato
surf = np.load('/scratch/fiorello/data_test/pores_deep_npy/00/surf_40.0.npy')
print("Forma:", surf.shape)

# Normalizza tra 0 e 1 per il salvataggio come immagine
#surf_norm = (surf - surf.min()) / (surf.max() - surf.min())

# Salva come PNG (scala di grigi)
plt.imsave('surf_400.png', surf, cmap='RdBu_r')
print("Salvato: surf_400.png")
