# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import numpy as np

from textwiser.transformations.nmf import _ScikitTransformation
from textwiser.utils import OutputType, convert

try:
    from umap import UMAP
    model = UMAP
except ModuleNotFoundError:
    model = None


class _UMAPTransformation(_ScikitTransformation):
    Model = model

    def __init__(self, model=None, deterministic_init: bool = False, **kwargs):
        super().__init__(model=model, **kwargs)
        self.deterministic_init = deterministic_init

    def _get_deterministic_init(self, x: np.ndarray):
        # By default, training a UMAP model in MacOS will lead to a different set of embeddings than in Ubuntu.
        # This is due to Numba having OS-based seeds
        # https://numba.pydata.org/numba-doc/latest/reference/pysupported.html
        # If we were developing UMAP, we could probably use np.random.seed within the JIT:
        # https://github.com/lmcinnes/umap/issues/153
        # Since we don't, the way we can control this is by doing the initialization manually
        # The initialization is taken directly from UMAP:
        # https://github.com/lmcinnes/umap/blob/623eb481a9037af8c982659227f22095a9e038b4/umap/umap_.py#L1074
        return np.random.default_rng(self.init_args['random_state'])\
            .uniform(low=-10.0, high=10.0, size=(x.shape[0], self.init_args['n_components']))

    def _fit(self, x, y=None):
        if self.deterministic_init:
            self.model = self.Model(**self.init_args, init=self._get_deterministic_init(x))
        else:
            self.model = self.Model(**self.init_args)
        self.model.fit(x, convert(y, OutputType.array))

    def _fit_transform(self, x, y=None):
        if self.deterministic_init:
            self.model = self.Model(**self.init_args, init=self._get_deterministic_init(x))
        else:
            self.model = self.Model(**self.init_args)
        return self.model.fit_transform(x, convert(y, OutputType.array)).astype(np.float32)

    @property
    def input_types(self):
        return OutputType.array,
