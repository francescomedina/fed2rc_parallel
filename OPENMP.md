# Fed2RC con OpenMP nativo

Questa variante parte dal commit
`209009c55b6b4d84b748ac5043ab48ee568fe4e5`. I due cicli numerici sono
implementati in C++17 e chiamati dal programma Python tramite pybind11.
Le direttive OpenMP sono nel codice utilizzato da Fed2RC, non soltanto in una
demo. La generazione dei parametri ROCKET resta quella di `sktime==1.1.0`:
usa internamente Numba, ma la nostra esecuzione parallela è OpenMP nativo.

## Compilazione e verifica

Servono Python, un compilatore C++17 con OpenMP, OpenBLAS con CBLAS e LAPACKE
(`cblas.h`, `lapacke.h` e relative librerie). Su macOS con Homebrew:

```bash
brew install libomp openblas
```

Su Linux usare GCC e i pacchetti di sviluppo OpenBLAS/LAPACKE della propria
distribuzione. `OPENBLAS_ROOT` e, con Apple Clang, `LIBOMP_ROOT` permettono di
specificare prefissi diversi da quelli standard. Tutte le librerie devono
usare la stessa ABI LAPACK con interi a 32 bit.
Su Linux il collegamento predefinito usa `lapacke openblas`; per una build
OpenBLAS che include già LAPACKE impostare `FED2RC_BLAS_LIBRARIES=openblas`
prima di compilare.

Dalla radice della repository, per installare il programma completo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python -m pip install -e .
```

Per compilare e verificare soltanto i kernel, senza installare il framework
federato Fluke e WandB, sostituire la riga con `requirements.txt` con:

```bash
python -m pip install numpy scipy scikit-learn pandas sktime==1.1.0 numba threadpoolctl pytest
```

Configurare i thread **prima** di avviare Python:

```bash
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
python -c 'from openmp_utils import native; print(native.runtime_info())'
python -m pytest -q
```

`default_threads` conserva la configurazione OpenMP all'importazione, prima
del limite BLAS, e viene applicato esplicitamente ai kernel: alcune build
OpenMP di OpenBLAS condividono lo stesso runtime e possono altrimenti ridurre
anche il parallelismo esterno a un solo thread. `probe_team_size` misura i
thread realmente creati in una piccola regione di prova, mentre
`runtime_max_threads` mostra il limite corrente, che può legittimamente
essere 1 dopo la configurazione delle BLAS.

Nei kernel il numero di worker viene limitato anche dal lavoro disponibile;
la regione di prova non applica questo limite. Nelle chiamate Python
`threads=0` usa `default_threads`, mentre `threads=1` o `threads=4` seleziona
esplicitamente il numero richiesto.

Per un'esecuzione completa servono inoltre i dataset UCR e la configurazione
del programma descritti nel README. I punti di ingresso rimangono
`code/fed2rc_no_fluke.py` e il modulo Fluke `code/fed2rc.py`: entrambi chiamano
i kernel nativi. Per una prova breve del programma, predisporre
`UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv` e `GunPoint_TEST.tsv` nella
radice della repository, quindi, con l'ambiente sopra configurato:

```bash
cd code
python fed2rc_no_fluke.py --debug --datasets GunPoint --experiment-seeds 1 -c 2 -k 32 -r 1
```

`--debug` evita l'accesso a WandB. `-r 1` conserva la semantica originale:
esegue i round 0 e 1. Il numero di client federati (`-c`) e il numero di
thread OpenMP sono parametri distinti. Le modifiche Slurm/Spack precedenti non sono incluse
in questa variante; non occorre accedere al cluster per compilarla e provarla.

Oltre ai kernel OpenMP, il programma comprende tre piccole modifiche d'uso:

- `--datasets` e `--experiment-seeds` selezionano prove ridotte, conservando
  i valori originali quando le opzioni sono omesse.
- `--debug` evita anche la precedente interrogazione iniziale dell'API WandB.
- `np.atleast_2d(model.coef_)[0]` mantiene compatibile l'estrazione dei pesi
  con la forma restituita dalle versioni recenti di scikit-learn.

## ROCKET: decomposizione del dominio e località

[`transform_features`](code/openmp_utils.py) normalizza ogni serie una volta,
con la precisione originale, poi converte in `float32`. La normalizzazione
passa da $O(KNT)$ a $O(NT)$, dove $N$ è il numero di serie, $T$ la loro
lunghezza e $K$ il numero di kernel. La convoluzione continua a dipendere
da tutti e tre i parametri.

[`rocket_map`](native/rocket.hpp) suddivide l'output in tile di 32 serie e
8 seed. `parallel for collapse(2) schedule(static)` distribuisce le coppie
di tile; `default(none)`, `shared` e `firstprivate` esplicitano lo stato.
Ogni cella dell'output ha un unico proprietario: è una **map data parallel**
senza dipendenze tra iterazioni e senza lock o atomiche. La barriera finale
garantisce che Python legga risultati completi.

Il blocco di serie viene riutilizzato con più kernel. Il risultato è scritto
direttamente in ordine per colonne, evitando l'array per ogni seed e la
successiva concatenazione. PPV e massimo vengono accumulati mentre si
calcola la convoluzione: non si materializza il vettore delle risposte.
Il percorso interno senza padding evita il controllo dei limiti per ogni
coefficiente. I tile riducono il traffico e favoriscono il riuso in cache;
le dimensioni scelte sono semplici costanti, non un autotuning per una CPU.

Non si promette l'assenza assoluta di **false sharing**: confini dei tile e
colonne non allineate possono condividere cache line. Non esistono però
contatori condivisi aggiornati ripetutamente. Il beneficio effettivo della
località va misurato sull'hardware di destinazione.

## Lambda: map–reduce e memoria privata

[`lambda_map`](native/solver.hpp) usa una regione `omp parallel` e un
`omp for schedule(static)`. Ogni iterazione risolve
$(A+\lambda_i I)W_i=b$ con LU generale e pivoting parziale LAPACK, come la
famiglia di solutori usata da `numpy.linalg.solve`. Non calcola l'inversa e
non sostituisce LU con Cholesky, che richiederebbe ipotesi ulteriori su $A$.

Ogni worker possiede una matrice LU, i termini noti/pesi, i pivot e un buffer
di predizione. I buffer sono allocati prima della regione parallela e
riutilizzati tra lambda; il worker esegue il primo accesso alle loro pagine.
Questa politica di **first touch** può favorire la località NUMA quando i
thread sono vincolati ai core; non è una garanzia portabile di collocazione.

Le matrici per LAPACK sono in ordine per colonne, senza trasposizioni per
ogni solve. La regolarizzazione modifica solo la diagonale della copia
privata. La predizione usa DGEMM su blocchi di 64 righe, consumati subito
per contare le classificazioni corrette: non viene costruito un temporaneo
$N\times C$ per ogni lambda. Per $P$ worker, $d$ caratteristiche e $C$ classi,
lo scratch principale in double è circa

$$8P(d^2+dC+64C)\ \text{byte},$$

oltre ai pivot, agli input condivisi e agli eventuali buffer interni BLAS.
Per esempio, con $d=1000$, $C=4$, $P=4$ sono circa 30,65 MiB di scratch
double. Aumentare i thread aumenta anche questa memoria privata.

Ogni iterazione scrive una sola accuratezza e uno stato nella propria cella.
Dopo la barriera, [`select_lambda`](code/openmp_utils.py) controlla gli errori
ed esegue una riduzione sequenziale $O(L)$, scegliendo il lambda maggiore a
parità di accuratezza. Le fattorizzazioni dense richiedono complessivamente
$O(Ld^3)$: la riduzione è normalmente marginale. Nessun massimo globale
viene aggiornato nel ciclo parallelo.

Le BLAS sono limitate a un thread per evitare **oversubscription** e
parallelismo annidato. Il limite è una politica del processo impostata
all'importazione, non un contesto temporaneo aperto e ripristinato da ogni
chiamata concorrente. Non modificarlo durante l'esecuzione dei kernel.

## GIL, cache e contratto degli input

[`bindings.cpp`](native/bindings.cpp) valida forme e valori con il GIL,
conserva i proprietari degli array, estrae i puntatori e usa
`py::gil_scoped_release` durante il calcolo nativo. I worker non chiamano
Python. Gli input devono rimanere invariati fino al ritorno della chiamata:
rilasciare il GIL non rende sicure modifiche concorrenti agli stessi buffer.

La sola cache è `lru_cache(maxsize=4096)` dei parametri deterministici dei
kernel, indicizzati da lunghezza della serie e seed. Contiene tuple
immutabili, indipendenti dal dataset. La gestione interna sincronizza i
metadati; due miss contemporanei possono rigenerare lo stesso kernel senza
corrompere la cache. Questa preparazione precede la regione OpenMP.

Non esiste una cache globale delle trasformazioni: una chiave basata solo
sul client potrebbe restituire dati obsoleti, mentre hash e copie complete
introdurrebbero traffico aggiuntivo. Il flag storico `caching` controlla
ora soltanto la cache dei parametri. La memoria di quest'ultima è limitata
indipendentemente da $N$, mentre un output PPV richiede $4NK$ byte: con
$N=10000$ e $K=1000$ sono 38,15 MiB. La riduzione dei ricalcoli tramite una
cache dei dataset richiederebbe un contratto esplicito sul loro ciclo di vita.

## Correttezza e prestazioni

[`tests/test_openmp.py`](tests/test_openmp.py) confronta le trasformazioni
con il codice estratto dal commit originale, e i lambda con il ciclo
NumPy originale. Verifica ordine e duplicati dei seed, precisioni e layout,
parità con uno/quattro thread, chiamate concorrenti, input immutati,
spareggi, errori dei solve e configurazione dei thread dopo il limite BLAS.
I test su GunPoint e ItalyPowerDemand usano gli split ufficiali: scelgono
lambda solo sul training e confrontano le predizioni sul test tenuto fuori.
Una regressione aggiuntiva confronta l'intero protocollo federato originale
e quello modificato su ArrowHead, con due client, 32 kernel e round 0 e 1:
verifica seed selezionati, predizioni e metriche finali con uno/quattro
thread. Queste prove non equivalgono a riprodurre tutti gli esperimenti
del progetto.

Il C++ non usa `fast-math`. ROCKET originale lo usa internamente: compilatori
e librerie BLAS differenti possono produrre arrotondamenti diversi. I test
richiedono PPV e predizioni identiche sui casi verificati, con tolleranza per
le feature del massimo. Non dimostrano equivalenza bit per bit su ogni input
né una garanzia statistica universale sull'accuratezza.

Per una misura locale riproducibile, dalla radice della repository:

```bash
python measure_openmp.py --output results/openmp_local.json --repeats 5 --threads 1 2 4
```

Lo script usa dati sintetici fissi: ROCKET con 256 serie di lunghezza 256 e
128 kernel; lambda con matrici di dimensione 128, tre classi, 256 campioni
di valutazione e 64 candidati. Prima dei tempi verifica PPV, vettore delle
accuratezze e lambda scelto contro l'originale; riscalda JIT, cache dei
parametri e regioni OpenMP. Le ripetizioni alternano i modi in ordine
mescolato deterministicamente e riportano tempi grezzi e mediane in JSON.
Il tempo include le intere funzioni, con validazione e adattamento dei
buffer; esclude import, generazione dei dati e riscaldamento. La cache dei
parametri nativi è calda, mentre l'originale ripete `fit` a ogni chiamata.
Le BLAS sono a un thread, ma NumPy e l'estensione possono usare librerie
BLAS diverse: backend e versioni sono riportati nel JSON. `native_speedup`
confronta lo stesso codice con uno/più thread; `original_over_native`
comprende anche queste differenze e le ottimizzazioni sequenziali.

Per misurare la **strong scalability**, mantenere fissi dataset, seed,
lambda e stato della cache e confrontare questo stesso kernel con uno e $P$
worker, dopo il riscaldamento:

$$S(P)=T_1/T_P,\qquad E(P)=S(P)/P.$$

La legge di Amdahl, con frazione sequenziale $s$, dà il limite ideale
$S(P)\leq1/[s+(1-s)/P]$. Preparazione dei kernel, normalizzazione, adattamento
dei buffer e resto del protocollo restano costi da includere nel tempo
end-to-end. Per la **weak scalability** aumentare il lavoro proporzionalmente
a $P$, per esempio $L$ mantenendo fisso $d$, e misurare
$E_{weak}(P)=T(1,W)/T(P,PW)$.

Il confronto con il commit originale misura insieme parallelismo e
ottimizzazioni sequenziali; non va chiamato puro speedup OpenMP. Per studiare
il traffico CPU–memoria, l'intensità aritmetica è $I=\mathrm{FLOP}/\mathrm{byte}$
e il modello Roofline pone $\mathrm{prestazioni}\leq\min(F_{peak}, B_{mem}I)$.
Eliminare temporanei può aumentare $I$, ma senza contatori hardware non si
può quantificare la riduzione del traffico verso la RAM. Nessuno speedup è
garantito prima della misura, soprattutto per input piccoli.

Riferimenti primari: [rilascio del GIL in pybind11](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil),
[OpenMP `collapse`](https://www.openmp.org/spec-html/5.2/openmpsu30.html),
[LAPACK DGESV](https://netlib.org/lapack/explore-html/d8/da6/group__gesv_ga831ce6a40e7fd16295752d18aed2d541.html).

## Verifica locale del 29 settembre 2026

Compilazione ed esecuzione con Apple Clang/OpenMP su Apple M1 Pro; verificata
anche la sintassi C++/OpenMP con GCC 15. Nessuna esecuzione su HPC4AI.
La suite completa ha superato **37 test**. Il driver standalone con GunPoint,
2 client, 32 kernel iniziali per client, seed sperimentale 1 e `-r 1` termina
con accuracy 0,98 e F1 0,97959. Questa singola prova non è una stima generale
della qualità predittiva; il confronto completo con l'originale è il test
su ArrowHead. Per i dati binari il driver include la correzione compatibile
con la forma moderna di `RidgeClassifierCV.coef_`, descritta sopra.

Misura sintetica dei due helper, mediana di cinque ripetizioni dopo warm-up:

| Helper | Riferimento Python | OpenMP 1 thread | OpenMP 4 thread | Speedup nativo 1→4 |
|---|---:|---:|---:|---:|
| Trasformazione ROCKET | 316,80 ms | 29,96 ms | 8,21 ms | 3,65× |
| Ricerca lambda | 13,96 ms | 8,62 ms | 2,55 ms | 3,38× |

I tempi del riferimento contro OpenMP includono anche il riuso dei parametri,
la normalizzazione unica, overhead Python e differenze BLAS. Solo l'ultima
colonna misura la strong scalability del medesimo codice nativo. Il report
[openmp_local.json](results/openmp_local.json) conserva dimensioni, versioni,
runtime, ordine delle prove e tutti i tempi grezzi. Non è una misura
end-to-end del protocollo né una previsione delle prestazioni Broadwell.
