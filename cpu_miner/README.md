Descriere:

- Directorul cpu_miner conține implementarea deja făcută pe CPU. Acesta NU face parte din rezolvarea temei. Scopul codului este de a înțelege funcționalitatea pe CPU, că apoi să optimizați căutarea nonce-ului pe GPU, folosind CUDA.
- Acesta conține 5 teste:
 - 4 pentru a vă familiariza cu funcțiile folosite.
 - al 5-lea este efectiv implementarea miner-ului pe CPU.
- Căutarea nonce-ului din testul 5 ar trebui să dureze ~2s pe xl, pentru o dificultate de 5 leading 0s.
- Aceasta este o abordare simplistă a calculării unui block hash, cu complexitate redusă. Nu reflectă implementarea reală a algoritmilor de consens POW, având scop pur educativ.

Pasi pentru rulare:

1. To compile:  make
2. To run:      make run
3. To clean:    make clean
