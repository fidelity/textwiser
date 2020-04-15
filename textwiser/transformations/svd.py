# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import torch

from textwiser.utils import device
from textwiser.transformations.base import _BaseTransformation


class _SVDTransformation(_BaseTransformation):
    def __init__(self, n_components=None, **kwargs):
        super(_SVDTransformation, self).__init__(wrap_list_input=True)
        self.V = None
        self.n_components = n_components
        self.init_args = kwargs

    def _fit(self, x, y=None):
        _, _, V = torch.svd(x, **self.init_args)
        if self.n_components:
            V = V[:, :self.n_components]
            if V.shape[1] < self.n_components:
                V = torch.cat((V, torch.zeros(V.shape[0], self.n_components - V.shape[1], requires_grad=True,
                                              device=device)), dim=1)
        self.V = V

    def _forward(self, x):
        return x @ self.V
