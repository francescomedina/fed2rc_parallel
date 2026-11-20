"""Optional two-stage backend: orthogonal equivalence, boundaries and ridge.

Without build_twostage.py this module is explicitly skipped.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from openmp_utils import select_lambda, transform_features

two = pytest.importorskip('_fed2rc_twostage')


@pytest.mark.parametrize('n', [1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 257])
@pytest.mark.parametrize('tile', [16, 32, 64])
@pytest.mark.parametrize('p', [1, 4])
def test_full_orthogonal_reduction_and_solve(n, tile, p):
    rng = np.random.default_rng(471+n)
    z = rng.normal(size=(n+7, n))
    A = np.asfortranarray(z.T@z)
    b = np.asfortranarray(rng.normal(size=(n, 3)))
    before = A.copy(), b.copy()
    d, e, g, Q, active = two._plan(A, b, np.eye(n), p, tile, min(16, tile))
    T = np.diag(d)+np.diag(e, 1)+np.diag(e, -1)
    assert np.linalg.norm(A-Q@T@Q.T)/np.linalg.norm(A) < 2e-13
    np.testing.assert_allclose(Q.T@Q, np.eye(n), rtol=0, atol=2e-13)
    np.testing.assert_allclose(g, Q.T@b, rtol=2e-12, atol=2e-13)
    for lam in (.001, 1.):
        W = Q@np.linalg.solve(T+lam*np.eye(n), g)
        shifted = A+lam*np.eye(n)
        assert np.linalg.norm(shifted@W-b)/np.linalg.norm(b) < 2e-11
        np.testing.assert_allclose(W, np.linalg.solve(shifted, b), rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(A, before[0])
    np.testing.assert_array_equal(b, before[1])
    if n>=16:
        assert active>=1  # Workers actually applying reflector blocks.


@pytest.mark.parametrize('n', [17, 65, 128, 257])
@pytest.mark.parametrize('rank', [8, 300])
def test_scores_projections_and_threads(n, rank):
    rng = np.random.default_rng(19+n)
    z = rng.normal(size=(rank, n))
    A, b = np.asfortranarray(z.T@z), np.asfortranarray(rng.normal(size=(n, 3)))
    X, y = rng.normal(size=(73, n)), rng.integers(3, size=73)
    ls = np.array([.001, .1, 1., .1])
    expected = select_lambda(A, b, X, y, ls, threads=1)
    for p in (1, 2, 4, 8):
        actual = select_lambda(A, b, X, y, ls, threads=p, solver='two_stage')
        assert actual[:2] == expected[:2]
        np.testing.assert_array_equal(actual[2], expected[2])
        # A rank-deficient matrix has no unique null-space basis: task ordering
        # can choose another valid Q on another call. Recover Q and XQ from
        # the SAME reduction rather than comparing unrelated coordinates.
        d, e, g, stacked, _ = two._plan(A, b, np.vstack((X, np.eye(n))), p, 32, 16)
        H, Q = stacked[:len(X)], stacked[len(X):]
        np.testing.assert_allclose(H, X@Q, rtol=2e-10, atol=2e-12)
        T = np.diag(d)+np.diag(e, 1)+np.diag(e, -1)
        assert np.linalg.norm(A-Q@T@Q.T)/np.linalg.norm(A) < 2e-13


@pytest.mark.parametrize('tile', [32, 64])
@pytest.mark.parametrize('p', [4, 8])
def test_repeated_wavefront_publication(tile, p):
    # On ARM the upstream relaxed atomic flags intermittently published task
    # completion before matrix/V/tau writes. Check every repeated reduction,
    # not just classification: small matrix errors can leave predictions intact.
    rng = np.random.default_rng(2026+17*128+13*256+3)
    z = rng.uniform(0, 1, size=(256, 128))
    A = np.asfortranarray(z.T@z)
    b = np.ones((128, 2), order='F')
    for _ in range(80):
        d, e, _, Q, _ = two._plan(A, b, np.eye(128), p, tile, 16)
        T = np.diag(d)+np.diag(e, 1)+np.diag(e, -1)
        assert np.linalg.norm(A-Q@T@Q.T)/np.linalg.norm(A) < 2e-13


def test_parallel_row_tasks_and_concurrency():
    n = 129
    A, b, X, y = np.eye(n, order='F'), np.zeros((n, 2), order='F'), np.ones((77, n)), np.zeros(77, np.int64)
    _, _, _, _, active = two._plan(A, b, X, 4, 32, 16)
    assert active == 4
    def run(i):
        return select_lambda(A, b, X, y, [.1, 1., .3, 1.], solver='two_stage', threads=1+i%4)
    with ThreadPoolExecutor(3) as pool:
        for result in pool.map(run, range(12)):
            assert result[:2] == (1., 1.)
            np.testing.assert_array_equal(result[2], np.ones(4))


@pytest.mark.parametrize('bad', [dict(bandwidth=3), dict(bandwidth=18), dict(inner_block=0),
                                 dict(inner_block=65), dict(threads=0)])
def test_invalid_controls(bad):
    with pytest.raises(ValueError):
        two._plan(np.eye(32, order='F'), np.ones((32, 2), order='F'), np.ones((5, 32)), **bad)


def test_invalid_data_and_indefinite_system():
    n = 33
    A, b, X, y = np.eye(n), np.ones((n, 2)), np.ones((5, n)), np.zeros(5, np.int64)
    for field, bad in [('A', np.full((n, n), np.nan)), ('y', np.full(5, 2)),
                        ('lambdas', [0]), ('lambdas', [np.inf]), ('y', np.zeros(5))]:
        values = dict(A=A, b=b, X=X, y=y, lambdas=[.1]); values[field] = bad
        with pytest.raises(ValueError): select_lambda(**values, solver='two_stage')
    A[0, 1] = .1
    with pytest.raises(ValueError, match='symmetric'):
        select_lambda(A, b, X, y, [.1], solver='two_stage')
    with pytest.raises(np.linalg.LinAlgError):
        select_lambda(-np.eye(n), b, X, y, [.1], solver='two_stage')


def test_default_stays_independent_of_optional_module(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, '_fed2rc_twostage', None)
    data = np.eye(17), np.zeros((17, 2)), np.ones((3, 17)), np.zeros(3, np.int64), [.1, 1.]
    assert select_lambda(*data, threads=1)[:2] == (1., 1.)
    with pytest.raises(ModuleNotFoundError):
        select_lambda(*data, threads=1, solver='two_stage')
    with pytest.raises(ValueError):
        select_lambda(*data, solver='unknown')


def test_real_rocket_search_and_final_predictions():
    from sktime.datasets import load_gunpoint
    from sklearn.preprocessing import LabelEncoder
    X, y = load_gunpoint(split='train', return_type='numpy3D')
    Xt, yt = load_gunpoint(split='test', return_type='numpy3D')
    encoder = LabelEncoder().fit(y)
    y = encoder.transform(y)
    Z = transform_features(X[:, 0, :], range(128), threads=4)
    Zt = transform_features(Xt[:, 0, :], range(128), threads=4)
    A, b = Z[:35].T@Z[:35], Z[:35].T@np.eye(2)[y[:35]]
    ls = np.logspace(-3, 0, 100)
    expected = select_lambda(A, b, Z[35:], y[35:], ls, threads=4)
    actual = select_lambda(A, b, Z[35:], y[35:], ls, threads=4, solver='two_stage')
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2])
    def predict(lam):
        W = np.linalg.solve(Z.T@Z+lam*np.eye(128), Z.T@np.eye(2)[y])
        W /= np.linalg.norm(W, axis=0)+1e-10
        return np.argmax(Zt@W, axis=1)
    np.testing.assert_array_equal(predict(actual[0]), predict(expected[0]))
