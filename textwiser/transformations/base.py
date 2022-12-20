# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import numpy as np
import torch

from textwiser.base import BaseFeaturizer
from textwiser.utils import convert, OutputType


class _BaseTransformation(BaseFeaturizer):
    def __init__(self, wrap_list_input=True):
        """Initializes a Transformation.

        Subclasses must call this __init__ method.

        Parameters
        ----------
        wrap_list_input : bool
            If true, any list input to fit, forward, or fit_transform functions will
            be stacked to a 2D tensor before the functions are called, and will be
            converted back to a list before being returned.
        """

        super(_BaseTransformation, self).__init__()
        self.wrap_list_input = wrap_list_input

    @property
    def input_types(self):
        return OutputType.tensor,

    def _check_input(self, x):
        if not isinstance(x, tuple(t.value for t in self.input_types)) or isinstance(x, np.matrix):
            return convert(x, self.input_types[0])
        return x

    def _forward(self, x):
        raise NotImplementedError("Transformations should implement the `_forward` method.")

    def fit(self, x, y=None):
        x = self._check_input(x)
        if self.wrap_list_input:
            if isinstance(x, list):  # happens after WordEmbedding
                x = torch.cat(x, 0)
        self._fit(x, y)

    def _wrap_list_input(self, fn, uses_y, x, y=None):
        sizes = None
        if isinstance(x, list):  # happens after WordEmbedding
            if len(x) == 0:
                return []
            sizes = [0]
            sizes.extend([doc.shape[0] for doc in x])
            x = torch.cat(x, 0)
        vec = fn(x, y) if uses_y else fn(x)
        if sizes:
            cs = np.cumsum(sizes)
            vec = [vec[cs[i]:cs[i + 1], :] for i in range(cs.shape[0] - 1)]
        return vec

    def forward(self, x):
        x = self._check_input(x)
        return self._wrap_list_input(self._forward, False, x) if self.wrap_list_input else self._forward(x)

    def fit_transform(self, x, y=None):
        x = self._check_input(x)
        return self._wrap_list_input(self._fit_transform, True, x, y) if self.wrap_list_input else self._fit_transform(x, y)

    def _fit_transform(self, x, y=None):
        x = self._check_input(x)
        self._fit(x, y)
        return self._forward(x)
