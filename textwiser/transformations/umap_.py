# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

from textwiser.transformations.nmf import _ScikitTransformation
from textwiser.utils import OutputType

try:
    from umap import UMAP
    model = UMAP
except ModuleNotFoundError:
    model = None


class _UMAPTransformation(_ScikitTransformation):
    Model = model

    @property
    def input_types(self):
        return OutputType.array,
