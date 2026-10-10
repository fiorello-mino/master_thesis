# Init per un quarto delle figure

15 init: 5 ellissoidi tondeggianti, 5 allungati e 5 coppie di sfere uguali collegate da cilindro. Tutte le figure sono orientate lungo z.

Dominio: x e y in [0,0.7], z in [-1.5,1.5]. I piani x=0 e y=0 tagliano le figure in quattro parti; nessun taglio lungo z. I centri sono sull’asse z che passa per (0,0,0).

Macro: `./macro/trench_14_14_60_quarter_shapes.3d`. End time: `0.15`. Output: `/scratch/fiorello/data3D/shapes/<nome_simulazione>`, con prefisso `quarter_` per distinguere questi casi da quelli completi.

Le dimensioni nelle chiavi ellipse->axis length sono i diametri delle figure complete. Il margine minimo di 0.1 è verificato sulle superfici geometriche rispetto a x=0.7, y=0.7 e z=±1.5; non riguarda la coda diffusa di phi. Raggi, semiassi, diametri e lunghezze sono tutti almeno 0.1.

Le altre impostazioni del template sono preservate, inclusi inner=1, outer=0, eps=0.1 e condizioni al bordo. La mesh e le condizioni di simmetria non sono state verificate nel simulatore. AMDiS non è stato eseguito.

## Ellissoidi

| Nome | Diametri x,y,z | Margine minimo esterno |
|---|---|---|
| quarter_ellipsoid_round_01 | [ 0.8, 0.8, 1 ] | 0.3 |
| quarter_ellipsoid_round_02 | [ 0.9, 0.9, 1.1 ] | 0.25 |
| quarter_ellipsoid_round_03 | [ 1, 1, 1.2 ] | 0.2 |
| quarter_ellipsoid_round_04 | [ 1.1, 1.1, 1.3 ] | 0.15 |
| quarter_ellipsoid_round_05 | [ 1.2, 1.2, 1.4 ] | 0.1 |
| quarter_ellipsoid_elongated_01 | [ 0.8, 0.8, 2 ] | 0.3 |
| quarter_ellipsoid_elongated_02 | [ 0.9, 0.9, 2.2 ] | 0.25 |
| quarter_ellipsoid_elongated_03 | [ 1, 1, 2.4 ] | 0.2 |
| quarter_ellipsoid_elongated_04 | [ 1.1, 1.1, 2.6 ] | 0.15 |
| quarter_ellipsoid_elongated_05 | [ 1.2, 1.2, 2.8 ] | 0.1 |

## Due sfere collegate

Centri delle sfere: (0,0,−L/2) e (0,0,+L/2). Il cilindro arriva ai centri delle sfere e si sovrappone a entrambe; è ruotato di 90° attorno a y per allineare il suo asse locale x a z.

| Nome | R sfere | r cilindro | L cilindro | Margine minimo esterno |
|---|---|---|---|---|
| quarter_two_spheres_cylinder_01 | 0.4 | 0.1 | 1.4 | 0.3 |
| quarter_two_spheres_cylinder_02 | 0.45 | 0.15 | 1.5 | 0.25 |
| quarter_two_spheres_cylinder_03 | 0.5 | 0.2 | 1.6 | 0.2 |
| quarter_two_spheres_cylinder_04 | 0.55 | 0.25 | 1.7 | 0.1 |
| quarter_two_spheres_cylinder_05 | 0.6 | 0.3 | 1.6 | 0.1 |
