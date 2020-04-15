# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

from sklearn.decomposition import LatentDirichletAllocation
from textwiser.transformations.nmf import _ScikitTransformation


class _LDATransformation(_ScikitTransformation):
    Model = LatentDirichletAllocation
