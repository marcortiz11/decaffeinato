
STOCHASTIC ROUNDING -> ROUND TO NEAREST

Per comprobar si realment el mètode d'arrodoniment stochastic rounding és millor que el round to Nearest,
entrenarem la xarxa amb els mateixos paràmetres i a partir de la epoch 20 canviarem el mètode d'arrodoniment.

Inicialment s'entrena una xarxa amb stochastic rounding que guardarà tots els paràmetres a la iteració 20.
Començem despŕes l'entrenament de una segona xarxa amb round to nearest usant aquests paràmetres.

La representació de dades és Fixed Point [2,14] (2 bits part entera i 14 bits part fraccionaria).


__________________

ROUND TO NEAREST -> STOCHASTIC ROUNDING

Com que Round to nearest és menys costós entrenarem les primeres 12 epochs amb el mètode d'arrodoniment round to nearest,
ja que hem observat que produeix molt poca degradació, i a partir de la 12 on els updates ja són més petits entrenarem amb stochastic rounding
per acabar de convergir.
