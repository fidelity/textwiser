# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import torch
from textwiser.options import PoolOptions
from textwiser.transformations.base import _BaseTransformation


def pool(x, pool_option: PoolOptions):
    if pool_option == PoolOptions.max:
        return torch.max(x, dim=0)[0]
    if pool_option == PoolOptions.min:
        return torch.min(x, dim=0)[0]
    if pool_option == PoolOptions.mean:
        return torch.mean(x, dim=0)
    if pool_option == PoolOptions.first:
        return x[0]
    if pool_option == PoolOptions.last:
        return x[-1]


class _PoolTransformation(_BaseTransformation):
    def __init__(self, pool_option=PoolOptions.max):
        super(_PoolTransformation, self).__init__(wrap_list_input=False)
        self.pool_option = pool_option

    def _fit(self, x, y=None):
        pass

    def _pool(self, x):
        return pool(x, self.pool_option)

    def _forward(self, x):
        return torch.stack([self._pool(sentence) for sentence in x], dim=0)
