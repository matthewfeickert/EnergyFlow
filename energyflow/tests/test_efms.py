from __future__ import absolute_import

import numpy as np
import pytest

import energyflow as ef
from test_utils import epsilon_percent, epsilon_diff

def test_has_efm():
    assert ef.EFM

def test_has_EFMSet():
    assert ef.EFMSet

def rec_outer(phat, v, q=None):
    q = phat if q is None else q
    return q if (q.ndim == v) else rec_outer(phat, v, q=np.multiply.outer(phat, q))

def slow_efm(zs, phats, v):
    return np.sum([z*rec_outer(phat, v) for z, phat in zip(zs, phats)], axis=0)

@pytest.mark.efm
@pytest.mark.parametrize('M', [1, 10, 50, 100, 1000])
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 1, 2, 'pf'])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
@pytest.mark.parametrize('nup', list(range(0,2)))
def test_efms(nup, measure, kappa, normed, M):
    if kappa == 'pf' and normed:
        pytest.skip('do not do pf with normed')

    events = ef.gen_random_events(2, M)
    e = ef.EFM(nup, measure=measure, kappa=kappa, normed=normed, coords='epxpypz')

    if kappa == 'pf':
        kappa = nup

    for event in events:
        if measure == 'hadrefm':
            zs = np.atleast_1d(ef.pts_from_p4s(event))
        elif measure == 'eeefm':
            zs = event[:,0]

        phats = event/zs[:,np.newaxis]
        zs = zs**kappa

        if normed:
            zs = zs/zs.sum()

        e_ans = e.compute(event)
        if nup == 0:
            assert epsilon_percent(e_ans, zs.sum(), 10**-13)
        else:
            s_ans = slow_efm(zs, phats, nup)
            assert epsilon_percent(s_ans, e_ans, 10**-13)

@pytest.mark.efm
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 1, 2, 'pf'])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
@pytest.mark.parametrize('M', [1, 10, 50, 100, 1000])
@pytest.mark.parametrize('sigs', [[(1,0),(1,1),(3,2),(0,4),(2,3),(1,2)],
                                  [(0,0),(1,0),(0,2),(1,2),(6,2),(1,5)]])
def test_efm_vs_efmset(sigs, M, measure, kappa, normed):
    if kappa == 'pf' and normed:
        pytest.skip('do not do pf with normed')

    efmset = ef.EFMSet(sigs, measure=measure, kappa=kappa, normed=normed, coords='epxpypz')
    efms = [ef.EFM(*sig, measure=measure, kappa=kappa, normed=normed, coords='epxpypz') for sig in sigs]

    for event in ef.gen_random_events(2, M):
        efm_dict = efmset.compute(event)
        for sig,efm in zip(sigs,efms):
            assert epsilon_percent(efm_dict[sig], efm.compute(event), 10**-12)

    