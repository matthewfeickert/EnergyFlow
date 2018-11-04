"""Base classes for EnergyFlow."""
from __future__ import absolute_import, division

from abc import ABCMeta, abstractmethod, abstractproperty
import multiprocessing
import sys

import numpy as np
from six import with_metaclass

from energyflow.measure import Measure


###############################################################################
# EFBase
###############################################################################
class EFBase(with_metaclass(ABCMeta, object)):

    @property
    def measure(self):
        return self._measure.measure

    @property
    def beta(self):
        return self._measure.beta

    @property
    def kappa(self):
        return self._measure.kappa

    @property
    def normed(self):
        return self._measure.normed

    @property
    def coords(self):
        return self._measure.coords

    @property
    def check_input(self):
        return self._measure.check_input

    @property
    def subslicing(self):
        return self._measure.subslicing


###############################################################################
# EFPBase
###############################################################################
class EFPBase(EFBase):

    def __init__(self, measure, beta, kappa, normed, coords, check_input):

        if 'efpm' in measure:
            raise ValueError('\'efpm\' no longer supported')

        self.use_efms = 'efm' in measure
        if self.use_efms and beta != 2:
            raise ValueError('Using an efm measure requires beta=2.')

        # store measure object
        self._measure = Measure(measure, beta, kappa, normed, coords, check_input)

    def get_zs_thetas_dict(self, event, zs, thetas):
        if event is not None:
            zs, thetas = self._measure.evaluate(event)
        elif zs is None or thetas is None:
            raise TypeError('If event is None then zs and thetas cannot also be None')
        return zs, {w: thetas**w for w in self._weight_set}

    def construct_efms(self, event, zs, phats):
        if event is not None:
            zs, phats = self._measure.evaluate(event)
        elif zs is None or phats is None:
            raise TypeError('If event is None then zs and thetas cannot also be None')
        return self._efmset.construct(zs=zs, phats=phats)

    @abstractproperty
    def _weight_set(self):
        pass

    @abstractproperty
    def _efmset(self):
        pass

    def _batch_compute_func(self, event):
        return self.compute(event, batch_call=True)

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def batch_compute(self, events, n_jobs=-1):
        """Computes the value of the EFP on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ 
            - The number of worker processes to use. A value of `-1` will attempt
            to use as many processes as there are CPUs on the machine.

        **Returns**

        - _1-d numpy.ndarray_
            - A vector of the EFP value for each event.
        """

        if n_jobs == -1:
            try: 
                self.n_jobs = multiprocessing.cpu_count()
            except:
                self.n_jobs = 4 # choose reasonable value

        # setup processor pool
        chunksize = max(len(events)//self.n_jobs, 1)
        if sys.version_info[0] == 3:
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = np.asarray(list(pool.imap(self._batch_compute_func, events, chunksize)))
        # Pool is not a context manager in python 2
        else:
            pool = multiprocessing.Pool(self.n_jobs)
            results = np.asarray(list(pool.imap(self._batch_compute_func, events, chunksize)))
            pool.close()

        return results


###############################################################################
# EFMBase
###############################################################################
class EFMBase(EFBase):

    def __init__(self, **kwargs):

        # verify correct measure
        if 'measure' in kwargs:
            assert 'efm' in kwargs['measure'], 'An EFM must use an efm measure'
        else:
            kwargs['measure'] = 'hadrefm'

        self._measure = Measure(**kwargs)

    @abstractmethod
    def compute(self, event=None, zs=None, phats=None):
        
        if event is not None:
            return self._measure.evaluate(event)
        elif zs is None or phats is None:
            raise ValueError('If event is None then zs and phats cannot be None.')
        return zs, phats

    def batch_compute(self, events, n_jobs=-1):
        """Computes the value of the EFP on several events.

        **Arguments**

        - **events** : array_like or `fastjet.PseudoJet`
            - The events as an array of arrays of particles in coordinates
            matching those anticipated by `coords`.
        - **n_jobs** : _int_ 
            - The number of worker processes to use. A value of `-1` will attempt
            to use as many processes as there are CPUs on the machine.

        **Returns**

        - _1-d numpy.ndarray_
            - A vector of the EFP value for each event.
        """

        if n_jobs == -1:
            try: 
                self.n_jobs = multiprocessing.cpu_count()
            except:
                self.n_jobs = 4 # choose reasonable value

        # setup processor pool
        chunksize = max(len(events)//self.n_jobs, 1)
        if sys.version_info[0] == 3:
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = np.asarray(list(pool.imap(self._batch_compute_func, events, chunksize)))
        # Pool is not a context manager in python 2
        else:
            pool = multiprocessing.Pool(self.n_jobs)
            results = np.asarray(list(pool.imap(self._batch_compute_func, events, chunksize)))
            pool.close()

        return results