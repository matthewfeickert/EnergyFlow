"""Implementations of the `EFM` and `EFMSet` classes and the `efp2efms` function."""
from __future__ import absolute_import, division

from collections import OrderedDict
from operator import itemgetter

import numpy as np

from energyflow.algorithms import einsum
from energyflow.efbase import EFBase
from energyflow.measure import measure_kwargs
from energyflow.utils import flat_metric, kwargs_check, timing
from energyflow.utils.graph_utils import *

__all__ = ['EFM', 'EFMSet', 'efp2efms']


###############################################################################
# EFM functions
###############################################################################

# allowed einsum symbols
I = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def efp2efms(graph):
    """This function converts an EFP to an EFM formula."""

    # build convenient data structure to hold graph information
    vds = get_valency_structure(graph)

    # dictionary to hold efm terms
    efms = {}

    # counter to store how to get fresh dummy indices
    ind = 0

    # iterate over vertices sorted by valency in decreasing order
    sorted_verts = sorted(valencies(graph).items(), key=itemgetter(1), reverse=True)
    for vert,valency in sorted_verts:

        # dict holding info for new efm term
        new_efm = {'upper_indices': '', 'lower_indices': ''}

        # iterate over neighboring vertices
        for neighbor,n_shared_edges in vds[vert].items():

            # if a new neighbor, assign fresh I
            if neighbor not in efms:
                new_I = I[ind:ind+n_shared_edges]
                ind += n_shared_edges
                new_efm['upper_indices'] += new_I

                # store I shared with that neighbor
                new_efm[neighbor] = new_I

            # if neighbor already has an efm factor, add already assigned indices to lower_indices
            else:
                new_efm['lower_indices'] += efms[neighbor][vert]

        # store new efm factor
        efms[vert] = new_efm

    einstr_list, efm_specs = [], []
    for vert,valency in sorted_verts:
        efm = efms[vert]
        lower_indices = efm['lower_indices']

        # conventionally put lowered indices before upper indices
        einstr_list.append(lower_indices + efm['upper_indices'])

        # add spec which is (nlow, nup) of efm
        nlow = len(lower_indices)
        efm_specs.append((nlow, valency - nlow))

    # return comma joined einstr and efm_specs
    return ','.join(einstr_list), efm_specs


###############################################################################
# EFM
###############################################################################

class EFM(object):

    """"""

    def __init__(self, nlow, nup, raw=False, rlfrom=None, subslicefrom=None):
        """A class representing a single EFM"""

        # store inputs
        self.nlow = nlow
        self.nup = nup
        self.raw = raw
        self.rlfrom = rlfrom
        self.subslicefrom = subslicefrom

        # get useful derived quantities
        self.v = self.nlow + self.nup
        self.sig = (self.nlow, self.nup)

        if self.raw:
            self.raw_einstr = ','.join([I[0]] + [I[0] + I[i+1] for i in range(self.v)])
            self.raw_einpath = ['einsum_path'] + [(0,1)]*self.v
            self.rl_diff = self.nlow
            self.rl_einstr = ','.join([I[:self.v]] + list(I[:self.nlow])) + '->' + I[:self.v]
            self.construct = self.raw_construct

        elif self.rlfrom is not None:
            if self.v != sum(self.rlfrom):
                raise ValueError('cannot raise/lower among different valency EFMs')
            diff = self.nlow - rlfrom[0]
            self.rl_diff = abs(diff)
            nlow_tup = (rlfrom[0], self.nlow) if diff > 0 else (self.nlow, rlfrom[0])
            self.rl_einstr = ','.join([I[:self.v]] + list(I[slice(*nlow_tup)])) + '->' + I[:self.v]
            self.construct = self.rl_construct
        elif self.subslicefrom is not None:
            num_up_subslices = self.subslicefrom[1] - self.nup
            num_low_subslices = self.subslicefrom[0] - self.nlow
            
            # perform check
            if num_up_subslices < 0 or num_low_subslices < 0:
                m = 'cannot perform subslicing from {} to {}'.format(self.subslicingfrom, self.sig)
                raise ValueError(m)

            self.subslice = tuple([0]*num_low_subslices + [Ellipsis] + [0]*num_up_subslices)
            self.construct = self.subslice_construct

    def raise_lower(self, tensor):

        # having optimize=False immediately turns this into pure c_einsum
        return np.einsum(self.rl_einstr, tensor, *[flat_metric(len(tensor))]*self.rl_diff, 
                         optimize=False)

    def raw_construct(self, zsphats):
        zs, phats = zsphats
        M, dim = phats.shape

        # if no lowering is needed
        if self.nlow == 0:
            self.data = np.einsum(self.raw_einstr, zs, *[phats]*self.v, optimize=self.raw_einpath)

        # lowering phats first is better
        elif M*dim < dim**self.v:
            low_phats = phats*flat_metric(dim)[np.newaxis]
            einsum_args = [low_phats]*self.nlow + [phats]*self.nup
            self.data = np.einsum(self.raw_einstr, zs, *einsum_args, optimize=self.raw_einpath)

        # lowering EFM is better    
        else:
            self.data = np.einsum(self.raw_einstr, zs, *[phats]*self.v, optimize=self.raw_einpath)
            self.data = self.raise_lower(self.data)

        return self.data

    def rl_construct(self, other_data):       
        self.data = self.raise_lower(other_data)
        return self.data

    def subslice_construct(self, other_data):
        self.data = other_data[self.subslice]
        return self.data

    def set_timer(self):
        self.times = []
        self.construct = timing(self, self.construct)


###############################################################################
# EFMSet
###############################################################################

class EFMSet(EFBase):

    """A class for holding a collection of `EFM`s and constructing them as a set."""

    def __init__(self, efm_specs, subslicing=False, **kwargs):
        """
        **Arguments**

        - **efm_specs** : {_list_, _tuple_, _set_} of _tuple_
            - A collection of tuples of length two specifying which EFMs this
            object is to hold. Each spec is of the form `(nlow, nup)` where these
            are the number of lower and upper indices that the EFM is to have.
        - **measure_dict** : _dict_
            - A dictionary of measure arguments to allow this EFMSet object to
            be called directly on events.
        - **subslicing** : _bool_
            - Whether the EFMs to be constructed have the subslicing property.
        """

        # store inputs
        self.subslicing = subslicing

        # verify no extra kwargs
        kwargs_check('EFMSet', kwargs, allowed=measure_kwargs)

        # verify correct measure
        if 'measure' in kwargs:
            assert 'efm' in kwargs['measure'], 'EFMSet must use an EFM measure'
        else:
            kwargs['measure'] = 'hadrefm'

        # setup measure through base class
        super(EFMSet, self).__init__(**kwargs)

        # get unique EFMs 
        self.unique_efms = frozenset(efm_specs)

        # setup EFMs based on whether we can subslice or not
        self.efms, self.efm_args, self.efm_rules = {}, {}, OrderedDict()
        if self.subslicing:
            self._subslicing_setup()
        else:
            self._full_setup()


    #################
    # PRIVATE METHODS
    #################

    def _find_subslice(self, sig):
        """Determine if sig can be subsliced from the EFMs currently stored in self.efms."""

        nlow, nup = sig
        bsigs = list(filter(lambda x: x[0] >= nlow and x[1] >= nup, self.efms))
        return min(bsigs, key=sum) if len(bsigs) else None

    def _find_minimum_rl(self, sig):
        """ """

        v = sum(sig)
        vsigs = list(filter(lambda x: sum(x) == v, self.efms))
        return min(vsigs, key=lambda x: abs(sig[0]-x[0]))

    def _subslicing_setup(self):
        """Setup the rules for constructing the EFMs using the fact that setting
        an index to zero "pops" it off, which is referred to as the subclicing 
        property. Typically, the EE measures have this property whereas the hadronic
        ones do not.
        """

        # ensure there is at least one EFM of each valency for rl purposes
        maxsig = max(self.unique_efms, key=sum) if len(self.unique_efms) else (0,0)
        self.unique_efms |= set((0,n) for n in range(1, sum(maxsig)+1))

        # sort EFMs to minimize raising/lowering operations
        # EFMs will be ordered first by decreasing v, then decreasing abs difference 
        # between nlow and nup, and then decreasing nup
        self.sorted_efms = sorted(self.unique_efms, key=itemgetter(1), reverse=True)
        self.sorted_efms.sort(key=lambda x: abs(x[0]-x[1]), reverse=True)
        self.sorted_efms.sort(key=sum, reverse=True)

        # take care of empty set
        if not len(self.sorted_efms):
            return

        # the first one must be raw constructed
        sig0 = self.sorted_efms[0]
        self.efms[sig0] = EFM(*sig0, raw=True)
        self.efm_args[sig0] = 'r'
        self.efm_rules[sig0] = 'constructing raw'

        for sig in self.sorted_efms[1:]:

            # determine if we can subslice
            big_sig = _find_subslice(sig, self.efms)
            if big_sig is not None:
                self.efms[sig] = EFM(*sig, subslicefrom=big_sig)
                self.efm_args[sig] = big_sig
                self.efm_rules[sig] = 'subslicing from {}'.format(big_sig)

            # find best raise/lower available
            else:
                rlsig = _find_minimum_rl(sig, self.efms)
                self.efms[sig] = EFM(*sig, rlfrom=rlsig)
                self.efm_args[sig] = rlsig
                rl_n = abs(rlsig[0]-sig[0])
                self.efm_rules[sig] = 'raising/lowering from {}, {}'.format(rlsig, rl_n)

    def _full_setup(self):
        """Setup the rules for constructing the EFMs without the assumption of any
        special properties.
        """

        # sort the EFMs first by increasing v and then by increasing nlow
        self.sorted_efms = sorted(self.unique_efms, key=itemgetter(0))
        self.sorted_efms.sort(key=sum)

        vprev, sigprev = None, None
        for sig in self.sorted_efms:
            v = sum(sig)

            # construct raw (all up) if this is a new valency
            if v != vprev:
                self.efms[sig] = EFM(*sig, raw=True)
                self.efm_args[sig] = 'r'
                self.efm_rules[sig] = 'constructing raw'

            # construct from lowering if we have a previous EFM with this v
            else:
                self.efms[sig] = EFM(*sig, rlfrom=sigprev)
                self.efm_args[sig] = sigprev
                self.efm_rules[sig] = 'lowering from {}'.format(sigprev)

            vprev, sigprev = v, sig


    #################
    # PUBLIC METHODS
    #################

    def compute(self, event=None, zs=None, phats=None):
        """Build the EFMs held by this EFMSet according to the predetermined strategy."""

        zsphats = super(EFMSet, self).compute(event, zs, phats)

        efm_dict = {}
        for sig in self.sorted_efms:
            arg = self.efm_args[sig]
            data_arg = zsphats if arg == 'r' else self.efms[arg].data
            efm_dict[sig] = self.efms[sig].construct(data_arg)
        return efm_dict

    def set_timers(self):
        for efm in self.efms.values():
            efm.set_timer()

    def get_times(self):
        return {sig: np.asarray(efm.times) for sig,efm in self.efms.items()}
