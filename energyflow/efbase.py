"""Base class for EnergyFlow classes"""
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
# EFBase
###############################################################################
class EFMBase(with_metaclass(ABCMeta, object)):

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
