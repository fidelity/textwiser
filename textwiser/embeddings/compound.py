# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import json
import os

from textwiser.base import BaseFeaturizer


class _CompoundEmbeddings(BaseFeaturizer):
    def __init__(self, schema, pretrained=None):
        super(_CompoundEmbeddings, self).__init__()
        self.schema = schema

    def _build_model(self):
        # To circumvent the circular dependency
        from textwiser.factory import get_document_embeddings

        schema = self.schema
        if isinstance(schema, str) and os.path.exists(schema):
            with open(schema, 'r') as fp:
                schema = json.load(fp)
        self.model = get_document_embeddings(schema)

    def fit(self, x, y=None):
        self._build_model()
        self.model.fit(x, y)

    def forward(self, x):
        return self.model(x)

    def fit_transform(self, x, y=None):
        self._build_model()
        return self.model.fit_transform(x, y=y)
