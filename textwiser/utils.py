# -*- coding: utf-8 -*-
# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

"""
:Author: FMR LLC

This module provides a number of constants and helper functions.
"""

from enum import Enum
import numpy as np
from scipy.sparse import csr_matrix, spmatrix
import torch
from typing import NamedTuple, NoReturn, List


class Constants(NamedTuple):
    """
    Constant values used by the modules.
    """

    kwargs = "kwargs"
    frozen = '_frozen'
    default_model = 'default'

    default_seed = 123456
    """The default random seed."""


class OutputType(Enum):
    array = np.ndarray
    sparse = spmatrix
    tensor = torch.Tensor

    @staticmethod
    def from_object(obj):
        for typ in OutputType.__members__.values():
            if isinstance(obj, typ.value):
                return typ
        raise ValueError("Not a valid type!")


def check_false(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is false, otherwise raises the given exception.
    """
    if expression:
        raise exception


def check_true(expression: bool, exception: Exception) -> NoReturn:
    """
    Checks that given expression is true, otherwise raises the given exception.
    """
    if not expression:
        raise exception


def split_tokenizer(document: str) -> List[str]:
    """Uses whitespace splitting to tokenize a document."""
    return document.split()


def convert(arr, typ: OutputType, dtype=None, detach=False):
    """Converts a numpy array/sparse matrix/torch tensor to the desired output type.

    Optionally converts them to the given dtype, and detaches a torch tensor from the computation graph.
    """
    if arr is None:
        return arr
    if isinstance(arr, list):
        return [convert(x, typ, dtype=dtype, detach=detach) for x in arr]
    if isinstance(arr, np.ndarray):
        # numpy to sparse
        if issubclass(typ.value, spmatrix):
            arr = csr_matrix(arr)
            if dtype is not None and arr.dtype != dtype:
                return arr.astype(dtype)
            return arr
        # numpy to torch
        elif issubclass(typ.value, torch.Tensor):
            arr = torch.from_numpy(arr).to(device)
            if dtype is not None and arr.dtype != dtype:
                return arr.type(dtype)
            return arr
        # numpy to numpy
        if dtype is not None and arr.dtype != dtype:
            return np.asarray(arr.astype(dtype))
        return np.asarray(arr)
    elif isinstance(arr, spmatrix):
        # sparse to numpy
        if issubclass(typ.value, np.ndarray):
            arr = np.array(arr.todense())
            if dtype is not None and arr.dtype != dtype:
                return np.asarray(arr.astype(dtype))
            return np.asarray(arr)
        # sparse to torch
        elif issubclass(typ.value, torch.Tensor):
            arr = torch.from_numpy(arr.todense()).to(device)
            if dtype is not None and arr.dtype != dtype:
                return arr.type(dtype)
            return arr
        # sparse to sparse
        if dtype is not None and arr.dtype != dtype:
            return arr.astype(dtype)
        return arr
    elif isinstance(arr, torch.Tensor):
        # torch to numpy
        if issubclass(typ.value, np.ndarray):
            arr = arr.detach().cpu().numpy()
            if dtype is not None and arr.dtype != dtype:
                return np.asarray(arr.astype(dtype))
            return np.asarray(arr)
        # torch to sparse
        elif issubclass(typ.value, spmatrix):
            arr = csr_matrix(arr.detach().cpu().numpy())
            if dtype is not None and arr.dtype != dtype:
                return arr.astype(dtype)
            return arr
        # torch to torch
        if dtype is not None and arr.dtype != dtype:
            return arr.type(dtype)
        return arr.detach() if detach else arr
    else:
        raise ValueError('Unsupported input type')


def set_params(obj, **kwargs):
    """Convenience method for allowing Scikit-learn style setting of parameters.

    This is especially useful when using the randomized or grid search model selection
    classes of Scikit-learn. See the `model_selection_example.ipynb` notebook for an example.
    """
    for k, v in kwargs.items():
        parts = k.split('__', 1)
        if len(parts) == 1:  # This is the object to modify
            if isinstance(obj, dict):  # Modifying the schema
                obj[k] = v
            elif isinstance(obj, (tuple, list)):
                if isinstance(obj[1], dict):  # Modifying the keyword arguments of a model name in the schema.
                    obj[1][k] = v
                else:  # Modifying a transform or concat in schema
                    obj[int(k)] = v
            elif hasattr(obj, '_set'):  # for _ArgBase setting
                obj._set(k, v)
            else:  # Modifying an actual object property
                setattr(obj, k, v)
        else:  # There is a sub-object that needs to be modified
            attr_name, key = parts
            if isinstance(obj, (dict, list)):  # Modifying the schema or a list of objects
                if attr_name.isnumeric():  # Indexing into a list
                    attr_name = int(attr_name)
                if isinstance(obj[attr_name], str):
                    # If the next object is a simple string name of a model, then convert it to a tuple of
                    # string name and an empty dictionary of keyword arguments.
                    obj[attr_name] = [obj[attr_name], dict()]
                set_params(obj[attr_name], **{key: v})
            else:
                set_params(getattr(obj, attr_name), **{key: v})


# A PyTorch device object that automatically detects if there is a GPU available
# If it is available, the TextWiser model will automatically be placed on a GPU
device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
