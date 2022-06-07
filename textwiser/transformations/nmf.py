# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import numpy as np
from sklearn.decomposition import NMF

from textwiser.transformations.base import _BaseTransformation
from textwiser.utils import convert, OutputType


class _ScikitTransformation(_BaseTransformation):
    Model = None

    def __init__(self, model=None, **kwargs):
        super(_ScikitTransformation, self).__init__(wrap_list_input=True)
        self.model = model
        self.init_args = kwargs

    @property
    def input_types(self):
        return OutputType.array, OutputType.sparse

    def _fit(self, x, y=None):
        self.model = self.Model(**self.init_args)
        self.model.fit(x, convert(y, OutputType.array))

    def _fit_transform(self, x, y=None):
        self.model = self.Model(**self.init_args)
        return self.model.fit_transform(x, convert(y, OutputType.array)).astype(np.float32)

    def _forward(self, x):
        return self.model.transform(x).astype(np.float32)


class _NMFTransformation(_ScikitTransformation):
    Model = NMF

    def _init_model(self, x: np.ndarray):
        if self.init_args['n_components'] <= min(x.shape[0], x.shape[1]) and 'init' not in self.init_args:
            # Use the old default from before sklearn 1.1 for backwards compat
            return self.Model(**self.init_args, init='nndsvd')
        return self.Model(**self.init_args)

    def _fit(self, x, y=None):
        self.model = self._init_model(x)
        self.model.fit(x, convert(y, OutputType.array))

    def _fit_transform(self, x, y=None):
        self.model = self._init_model(x)
        return self.model.fit_transform(x, convert(y, OutputType.array)).astype(np.float32)
