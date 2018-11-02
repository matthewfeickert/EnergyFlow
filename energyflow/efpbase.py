"""Base and helper classes for EFPs."""
from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Counter
import multiprocessing
import sys

import numpy as np
from six import with_metaclass

from energyflow.algorithms import einsum
from energyflow.measure import Measure
from energyflow.utils import timing, transfer

###############################################################################
# EFPBase
###############################################################################
class EFPBase(with_metaclass(ABCMeta, object)):

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
        return self._efmset.construct(zs, phats)

    @abstractproperty
    def _weight_set(self):
        pass

    @abstractproperty
    def _efmset(self):
        pass

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
# EFPElem
###############################################################################
class EFPElem(object):

    # if weights are given, edges are assumed to be simple 
    def __init__(self, edges, weights=None, einstr=None, einpath=None, k=None,
                              efm_einstr=None, efm_einpath=None, efm_spec=None):

        transfer(self, locals(), ['einstr', 'einpath', 'k', 'efm_einstr', 'efm_einpath', 'efm_spec'])

        self.process_edges(edges, weights)

        self.pow2d = 2**self.d
        self.ndk = (self.n, self.d, self.k)

        self.use_efms = self.efm_spec is not None
        if self.use_efms:
            self.efm_spec_set = frozenset(self.efm_spec)

    def process_edges(self, edges, weights):

        # deal with arbitrary vertex labels
        vertex_set = frozenset(v for edge in edges for v in edge)
        vertices = {v: i for i,v in enumerate(vertex_set)}
        
        # determine number of vertices, empty edges are interpretted as graph with one vertex
        self.n = len(vertices) if len(vertices) > 0 else 1

        # construct new edges with remapped vertices
        self.edges = [tuple(vertices[v] for v in sorted(edge)) for edge in edges]

        # get weights
        if weights is None:
            self.simple_edges = list(frozenset(self.edges))
            counts = Counter(self.edges)
            self.weights = tuple(counts[edge] for edge in self.simple_edges)

            # invalidate einsum quantities because edges got reordered
            self.einstr = self.einpath = None
        else:
            if len(weights) != len(self.edges):
                raise ValueError('length of weights is not number of edges')
            self.simple_edges = self.edges
            self.weights = tuple(weights)
        self.edges = [e for w,e in zip(self.weights, self.simple_edges) for i in range(w)]

        self.e = len(self.simple_edges)
        self.d = sum(self.weights)
        self.weight_set = frozenset(self.weights)

    def efp_compute(self, zs, thetas_dict):
        einsum_args = [thetas_dict[w] for w in self.weights] + self.n*[zs]
        return einsum(self.einstr, *einsum_args, optimize=self.einpath)

    def efm_compute(self, efms_dict):
        einsum_args = [efms_dict[sig] for sig in self.efm_spec]
        return self.pow2d * einsum(self.efm_einstr, *einsum_args, optimize=self.efm_einpath)

    def compute(self, zs, thetas_dict, efms_dict):
        return self.efm_compute(efms_dict) if self.use_efms else self.efp_compute(zs, thetas_dict)

    def set_timer(self):
        self.times = []
        self.compute = timing(self, self.compute)

    # properties set above:
    #     n, e, d, k, ndk, edges, simple_edges, weights, weight_set, einstr, einpath,
    #     efm_einstr, efm_einpath, efm_spec
