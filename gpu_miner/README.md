Descriere:

- Veți porni de la directorul gpu_miner, în care veți realiza implementarea în CUDA a logicii din cpu_miner.
- Veți implementa funcția device findNonce, care va paraleliza căutarea nonce-ului, folosind CUDA Threads. Aceasta trebuie implementată astfel încât să caute prin toate numerele de la 1 la MAX_NONCE.
- Pentru a va ajută, aveți deja implementate funcții ajutătoare în utils.cu. Vă recomandăm să va folosiți de ele în implementarea voastră.
Nonce-ul găsit, hash-ul block-ului, precum și timpul rulării kernel-ului, vor fi scrise într-un fișier results.csv, în urmă apelarii funcției printResult din utils.cu.

Observații:

1. Compilarea si rularea temei se vor face EXCLUSIV pe coada xl. Nu este nevoie să modificăți nimic în Makefile, regulile sunt deja făcute. Tot ce trebuie să faceți este să apelați make, make run și make clean, ca la laborator.
2. Veți modifica DOAR fișierul gpu_miner.cu. Celelate fișiere din directorul gpu_miner se vor suprascrie la testarea automată.
3. Deși nonce-ul este un număr întreg, potiziv, pe 32 biți, MAX_NONCE pe GPU este setat cu valoarea 1e8, în loc de UINT32_MAX (~4.29 * 1e9). Motivul este de a reduce timpul și de a nu întâmpină bottleneck-uri când sunt trimise multe job-uri, în același timp, de la mai mulți studenți, pe coada xl.

Pași pentru rulare:

1. To compile:  make
2. To run:      make run
3. To clean:    make clean
