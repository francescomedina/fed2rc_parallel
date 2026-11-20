"""Adattatore Python: prepara i dati e chiama i cicli OpenMP in native/*.hpp.

Percorso: Python -> bindings.cpp -> rocket.hpp / solver.hpp -> risultati Python.
Gli input non devono essere modificati da altri thread durante una chiamata.
La cache conserva solo i piccoli parametri dei filtri, indipendenti dal dataset:
memorizzare le feature richiederebbe hash/copie dei dati o un contratto esplicito
di immutabilita'. Numba resta interno a sktime.fit; i nostri cicli sono OpenMP.
"""
from functools import lru_cache

import numpy as np
from sktime.transformations.panel.rocket import Rocket
from threadpoolctl import threadpool_limits
from temporal import resolve_temporal_chunks, validate_temporal_chunks

import _fed2rc_openmp as native

# Il parallelismo e' gestito dai nostri worker OpenMP. Ogni chiamata BLAS usa
# un solo thread, evitando di moltiplicare i team (oversubscription).
# Impostiamo questa politica per tutto il processo dopo l'import nativo:
# cambiarla e ripristinarla a ogni chiamata creerebbe conflitti tra chiamate
# concorrenti. Non va modificata mentre i calcoli nativi sono in corso.
_BLAS_LIMIT = threadpool_limits(limits=1, user_api="blas")


@lru_cache(maxsize=4096)
def _kernel(timepoints, seed):
    """Genera un filtro con sktime e conserva in cache i suoi parametri immutabili."""
    rocket = Rocket(num_kernels=1, random_state=seed, n_jobs=1)
    rocket.fit(np.zeros((1, 1, timepoints)))
    weights, lengths, biases, dilations, paddings, _, _ = rocket.kernels
    # La chiave della cache e' (lunghezza della serie, seed): lo stesso seed
    # riproduce il filtro per quella lunghezza. Le tuple non sono modificabili.
    # lru_cache protegge i metadati, ma due richieste simultanee non in cache
    # possono rigenerare lo stesso filtro: e' lavoro duplicato, non corruzione.
    return (tuple(float(w) for w in weights), int(lengths[0]), float(biases[0]),
            int(dilations[0]), int(paddings[0]))


def transform_features(X, seeds, *, ppv_only=True, caching=True, threads=0,
                       temporal_chunks=1):
    """Estrae feature ROCKET nell'ordine dei seed; output float32 per colonne.

    Normalizziamo le N serie di lunghezza T una sola volta: costo O(N*T).
    caching riguarda solo i parametri dei filtri; ogni chiamata rilegge X.
    threads=0 usa la configurazione OpenMP salvata all'importazione.
    temporal_chunks=1 mantiene il percorso originale; 2/4/N suddividono le
    posizioni di output. 'auto' divide solo convoluzioni lunghe con pochi job.
    """
    temporal_chunks = validate_temporal_chunks(temporal_chunks)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("Expected (samples > 0, timepoints >= 2)")
    if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
        raise ValueError("ROCKET requires finite real input")
    seeds = [int(s) for s in seeds]
    if any(s < 0 or s > 2**31 - 1 for s in seeds):
        raise ValueError("Seeds must fit a nonnegative int32")
    # Ogni serie viene centrata e scalata una volta, prima di tutti i filtri.
    # Manteniamo precisione ed epsilon di Rocket._transform e convertiamo dopo.
    # np.require prepara float32 contigui per righe (C) e allineati (A).
    normalized = np.require(
        (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8),
        dtype=np.float32, requirements=["C", "A"],
    )
    k = len(seeds)
    weights = np.zeros((k, 11), dtype=np.float32)
    lengths = np.empty(k, dtype=np.int32)
    biases = np.empty(k, dtype=np.float32)
    dilations = np.empty(k, dtype=np.int32)
    paddings = np.empty(k, dtype=np.int32)
    # Prepariamo i parametri dei filtri in Python, conservando ordine e duplicati
    # dei seed. La cache evita solo di rigenerare parametri gia' noti.
    prepare = _kernel if caching else _kernel.__wrapped__
    for i, seed in enumerate(seeds):
        w, length, bias, dilation, padding = prepare(X.shape[1], seed)
        weights[i, :length] = w
        lengths[i], biases[i], dilations[i], paddings[i] = length, bias, dilation, padding
    jobs = max(1, ((len(X) + 31) // 32) * ((k + 7) // 8))
    chunks = resolve_temporal_chunks(
        temporal_chunks, X.shape[1], lengths, dilations, paddings,
        threads or native.default_threads, jobs, samples=len(X),
    )
    # Un'unica chiamata C++ per tutti i filtri: il lavoro viene poi suddiviso
    # in blocchi tra i thread dentro rocket.hpp.
    return native.transform(normalized, weights, biases, lengths, dilations, paddings,
                            ppv_only, threads, chunks)


def select_lambda(A, b, X, y, lambdas, *, threads=0, solver="lu"):
    """Valuta i lambda in parallelo e sceglie il migliore dopo la sincronizzazione.

    Prepariamo A/b per colonne (F), adatte a LAPACK; X resta per righe (C).
    Se tipo, layout e allineamento sono gia' compatibili, np.require non copia.
    Ogni worker C++ riusa i propri buffer per i candidati che gli sono assegnati.
    solver='two_stage' usa la riduzione PLASMA opzionale (build_twostage.py).
    """
    A, b = [np.require(a, dtype=np.float64, requirements=["F", "A"]) for a in (A, b)]
    X = np.require(X, dtype=np.float64, requirements=["C", "A"])
    y = np.asarray(y)
    if y.dtype.kind not in "iu" or y.ndim != 1:
        raise ValueError("y must be a vector of integer class indices")
    y = np.require(y, dtype=np.int64, requirements=["C", "A"])
    lambdas = np.require(lambdas, dtype=np.float64, requirements=["C", "A"])
    # Fase map: C++ risolve e valuta un sistema per ogni candidato lambda.
    # La chiamata torna quando tutti i worker hanno completato i loro risultati.
    if solver == "two_stage":
        import _fed2rc_twostage
        if threads < 0:
            raise ValueError("threads must be nonnegative")
        scores, status = _fed2rc_twostage.lambda_scores(
            A, b, X, y, lambdas, threads or native.default_threads)
    elif solver == "lu":
        scores, status = native.lambda_scores(A, b, X, y, lambdas, threads)
    else:
        raise ValueError("solver must be lu or two_stage")
    failed = np.flatnonzero(status)
    if len(failed):
        i = int(failed[0])
        raise np.linalg.LinAlgError(f"Lambda index {i}: solve/prediction failed (status {status[i]})")
    # Fase reduce: scansione sequenziale O(L) delle L accuratezze, dopo la barriera.
    # Non aggiorniamo un massimo condiviso durante il costoso calcolo parallelo.
    # A parita' di accuratezza scegliamo il lambda maggiore, indipendentemente
    # dall'ordine in cui i thread hanno terminato: spareggio deterministico.
    best = max(range(len(lambdas)), key=lambda i: (scores[i], lambdas[i]))
    return float(lambdas[best]), float(scores[best]), scores
