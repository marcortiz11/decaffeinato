 En aquest seguit de proves volem observar el comportament d'usar un altre tipus de representació de les dades: floating point.
 Es simularàn un altre cop tots els experiments anteriors per a veure quins son els resultats respecte l'altre representació de dades usada: fixed point.
 
 En aquest experiment el el fixed point tindrà una representació estàndar per a poder córrer en un futur amb processadors eficientment:
    ·1 bit signe
    ·5 bits exponent
    ·10 bits mantissa
    
----------#EN TOTAL 16 bits!#------------


 Experiments:
 1 - Aplicar tots els arrodoniments disponibles (Stochastic rounding, Round to nearest, with fixed probability) per a aquest nou tipus de dades.
 ...
 
 
 Finalment s'agafarà la millor representació de fixed point (ja que teniem diferents representacions) i la compararem amb l'execució del float de 16 bits (representació estàndar).
