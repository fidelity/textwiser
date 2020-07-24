# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import torch

from textwiser.utils import device
from textwiser.transformations.base import _BaseTransformation


class _SVDTransformation(_BaseTransformation):
    def __init__(self, n_components=None, **kwargs):
        super(_SVDTransformation, self).__init__(wrap_list_input=True)
        self.n_components = n_components
        self.init_args = kwargs

    def _fit(self, x, y=None):
        _, _, V = torch.svd(x, **self.init_args)
        if self.n_components:
            V = V[:, :self.n_components]
            if V.shape[1] < self.n_components:
                V = torch.cat((V, torch.zeros(V.shape[0], self.n_components - V.shape[1], requires_grad=True,
                                              device=device)), dim=1)
        self.V = torch.nn.Parameter(V)

    def _forward(self, x):
        return x @ self.V

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Overridden PyTorch state dict loading for setting the V param."""
        v_arg = prefix + 'V'
        if v_arg in state_dict:
            self.V = torch.nn.Parameter(state_dict[v_arg])
            state_dict.pop(v_arg)
        super(_SVDTransformation, self)._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
