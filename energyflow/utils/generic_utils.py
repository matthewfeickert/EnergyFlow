from __future__ import absolute_import, division, print_function

from functools import wraps
from itertools import repeat
import os
import sys
import time

import numpy as np

__all__ = [
    'default_efp_file', 
    'concat_specs', 
    'iter_or_rep',
    'kwargs_check',
    'timing', 
    'transfer'
]

# get access to the data directory of the installed package and the default efp file
ef_data_dir = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
default_efp_file = os.path.join(ef_data_dir, 'efps_d_le_10.npz')

# handle pickling methods in python 2
if sys.version_info[0] == 2:
    import copy_reg
    import types

    def pickle_method(method):
        func_name = method.__name__
        obj = method.__self__
        cls = obj.__class__
        return unpickle_method, (func_name, obj, cls)

    def unpickle_method(func_name, obj, cls):
        for cls in cls.mro():
            try:
                func = cls.__dict__[func_name]
            except KeyError:
                pass
            else:
                break
        return func.__get__(obj, cls)

    copy_reg.pickle(types.MethodType, pickle_method, unpickle_method)

# concatenates con. and disc. specs along axis 0, handling empty disc. specs
def concat_specs(c_specs, d_specs):
    if len(d_specs):
        return np.concatenate((c_specs, d_specs), axis=0)
    else:
        return c_specs

# return argument if iterable else make repeat generator
def iter_or_rep(arg):
    if isinstance(arg, (tuple, list)):
        if len(arg) == 1:
            return repeat(arg[0])
        else:
            return arg
    else:
        return repeat(arg)

# raises TypeError if unexpected keyword left in kwargs
def kwargs_check(name, kwargs, allowed=[]):
    for k in kwargs:
        if k not in allowed:
            raise TypeError(name + '() got an unexpected keyword argument \'{}\''.format(k))

# timing meta-decorator
def timing(obj, func):
    @wraps(func)
    def decorated(*args, **kwargs):
        ts = time.process_time()
        r = func(*args, **kwargs)
        te = time.process_time()
        obj.times.append(te - ts)
        return r
    return decorated

# transfers attrs from obj2 (dict or object) to obj1
def transfer(obj1, obj2, attrs):
    for attr in attrs:
        setattr(obj1, attr, obj2[attr] if isinstance(obj2, dict) else getattr(obj2, attr))
