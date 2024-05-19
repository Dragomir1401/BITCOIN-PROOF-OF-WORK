Nume: Marinescu Alexandru
Grupă: 332CA

Tema  Implementarea CUDA a algoritmului de consens Proof of Work din cadrul Bitcoin


____________________________________________Explicație pentru soluția aleasă:

    Intreg enuntul a fost implementat si functionalitatea este completa. Solutia aleasa 
utilizeaza cuda threads pentru gasirea nonce-ului foarte rapid, cu timpi mult sub o secunda.
Implementarea practic se bazeaza pe optimizarea pe gpu a variantei secventiale pe cpu.

    Toate bufferele si variabile sunt intai alocate folosind cudaMalloc in main pentru
a sustine rolul de device memory. Apoi memoria este initializata si se copiaza hashul
bloculu anterior in blocul curent.

    Apoi se initializeaza block sizeul si gridul ca fiind:

    block size = 256
    grid size = (n + block size.x - 1) / block size.x

    Formula pentru grid size este folosita pentru a asigura ca toate threadurile sunt
folosite.

    Kernelul este implementat dupa doua idei principale:

    - Copierea locala a contentului si creara local a hashului ca se ne asiguram ca threadurile
    nu acceseaza acelasi pointer in acelasi timp, rezultand intr-un race condition.

    - Asigurarea ca gasim doar primul hash si nonce corespunzator se face printr-un flag boolean
    global. Cand threadul intra in kernel, se verifica ca nu cumva flagul sa fi fost setat pe true
    de alt thread inainte, astfel se opreste totul la primul nonce corect.

    Apoi, cand se intoarce cu rezultatele in main, in host practic, acolo se copiaza datele din device
memory inapoi in host memory si se printeaza.

    In final, se elibereaza memoria alocata pentru device memory.

    Dificultatile au aparut in testarea temei pe cluster si testarea in general de fapt a kernelului.
Functii ca printf se comporta ciudat pe device memory si nu se pot folosi intr-un mod reliable pentru
debugging. De asemenea, timpul de asteptare pe cluster devine inconvenabil pentru debugging. Varianta
mai usoara a fost sa iau feedback de la functiile apelate prin cudaError_t si sa ma asigur ca totul
ruleaza corect.

_____________________________________________________________________Comentarii:

    Implementarea functioneaza cum trebuie si rezolva cu timp sub o secunda de fiecare data
problema folosind clusterul. Hashul gasit este:
    00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee
iar nonce-ul gasit este:
    515800. Hashingul a fost testat cu un tool online si este valid.

    Implementarea este eficienta si nu am intampinat probleme in implementare. S-ar putea folosind
operatii atomice sau alte tipuri de verificari pentru thread safety, dar aceasta implementare este
relativ suficienta in a rezolva problema. Poate pot exista edge cases cand un thread ar trece deja de verificarea
de la inceput si ar continua sa ruleza chiar si dupa aflarea rezulatatului de alt thread, dar din testele facute
asta nu s-a intamplat.


_______________________________________________________________________Feedback:

    Tema este utila si pentru aflarea a cateva notiuni legate de algoritmul de consens Proof of Work, dar
si in general ca exercitiu relativ scurt de implementare pe CUDA. Dificultatea a fost una redusa si nu a 
adus multe shimbari fata de un laborator obisnuit de CUDA. A fost mai complicata testarea temei decat implementarea
per total.


_______________________________________________________________________Resurse:

TODO
