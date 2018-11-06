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
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('kappa', [0, 1, 2, 'pf'])
@pytest.mark.parametrize('measure', ['hadrefm', 'eeefm'])
@pytest.mark.parametrize('nup', list(range(0,2)))
def test_efms(nup, measure, kappa, normed):

    if kappa == 'pf' and normed:
        pytest.skip('do not do pf with normed')

    events = ef.gen_random_events(5, 25)
    e = ef.EFM(nup, measure=measure, kappa=kappa, normed=normed, coords='epxpypz')

    if kappa == 'pf':
        kappa = nup

    for event in events:
        if measure == 'hadrefm':
            zs = ef.pts_from_p4s(event)
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
