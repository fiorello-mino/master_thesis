# Geometrie bulk: dominio 0.7 × 0.7 × 3.0 centrato in (0,0,0)

Sono presenti tutti i 15 init: 5 ellissoidi tondeggianti, 5 allungati e 5 coppie di sfere uguali collegate da un cilindro. I parametri geometrici sono anche in geometries.json. Le coppie usano la sintassi confermata dall’utente: `ellipse + ellipse1 + cylinder`, con parametri separati per ogni istanza.

Macro file impostato in tutti gli init: `./trench_14_14_60_centered.3d`. Il dominio è quello indicato dall’utente; il file mesh non è stato ispezionato né è stato eseguito AMDiS. La verifica dei margini riguarda le superfici geometriche, non la coda del campo di fase diffuso.

Piano e gaussiana rimossi. inner=1, outer=0 e gli altri parametri del template, compreso eps=0.1, sono mantenuti. I colli hanno raggio 0.10–0.12: eps=0.1 è comparabile a questi raggi, quindi la risoluzione delle interfacce richiede attenzione prima delle simulazioni di pinch-off.

## Ellissoidi

| Caso | Diametri x, y, z | Margine minimo |
|---|---|---|
| ellipsoid_round_01 | [ 0.28, 0.28, 0.32 ] | 0.210 |
| ellipsoid_round_02 | [ 0.32, 0.32, 0.38 ] | 0.190 |
| ellipsoid_round_03 | [ 0.36, 0.36, 0.42 ] | 0.170 |
| ellipsoid_round_04 | [ 0.4, 0.4, 0.46 ] | 0.150 |
| ellipsoid_round_05 | [ 0.44, 0.44, 0.5 ] | 0.130 |
| ellipsoid_elongated_01 | [ 0.28, 0.28, 1 ] | 0.210 |
| ellipsoid_elongated_02 | [ 0.32, 0.32, 1.4 ] | 0.190 |
| ellipsoid_elongated_03 | [ 0.36, 0.36, 1.8 ] | 0.170 |
| ellipsoid_elongated_04 | [ 0.42, 0.42, 2.2 ] | 0.140 |
| ellipsoid_elongated_05 | [ 0.48, 0.48, 2.6 ] | 0.110 |

## Due sfere uguali collegate lungo z

In ogni caso le sfere hanno lo stesso raggio R. I centri sono (0,0,−L/2) e (0,0,+L/2). Il cilindro è centrato nell’origine, ha lunghezza L e raggio r. In questo modo i suoi estremi sono nei centri delle sfere, con sovrapposizione reale. Il cilindro nel codice ha asse locale x: si imposta una rotazione di 90 gradi attorno a y per allinearlo a z. Tutte le figure sono centrate globalmente nell’origine.

| Caso | R sfere | r cilindro | L cilindro | Margine minimo |
|---|---|---|---|---|
| two_spheres_cylinder_01 | 0.16 | 0.10 | 0.60 | 0.190 |
| two_spheres_cylinder_02 | 0.18 | 0.10 | 0.90 | 0.170 |
| two_spheres_cylinder_03 | 0.20 | 0.10 | 1.20 | 0.150 |
| two_spheres_cylinder_04 | 0.22 | 0.12 | 1.60 | 0.130 |
| two_spheres_cylinder_05 | 0.24 | 0.10 | 2.00 | 0.110 |

## Directory di output

Ogni init usa `/scratch/fiorello/data3D/shapes/<nome_simulazione>`. Per esempio: `/scratch/fiorello/data3D/shapes/two_spheres_cylinder_01`. Questi sono percorsi per il computer che esegue AMDiS; qui sono stati scritti soltanto negli init.

Tutti i raggi, semiasse e lunghezze delle figure sono almeno 0.1. I centri e le componenti dei vettori di rotazione possono contenere zero o valori negativi: non sono dimensioni geometriche.
