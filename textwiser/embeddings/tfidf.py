# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from textwiser.base import BaseFeaturizer
from textwiser.utils import convert, Constants, OutputType


class _ScikitEmbeddings(BaseFeaturizer):
    Model = None

    def __init__(self, pretrained=None, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.vectorizer = None
        self.init_args = kwargs

    def _init_vectorizer(self):
        """Initializes the scikit learn vectorizer and returns if it needs to be fitted"""
        if isinstance(self.pretrained, str):
            if self.pretrained is Constants.default_model:
                self.vectorizer = self.Model(**self.init_args, dtype=np.float32)
                return True
            elif os.path.exists(self.pretrained):
                with open(self.pretrained, 'rb') as fp:
                    self.vectorizer = pickle.load(fp)
                    return False
            else:
                raise ValueError("Cannot find the specified path for pretrained model.")
        elif hasattr(self.pretrained, 'read'):  # File-like
            self.vectorizer = pickle.load(self.pretrained)
            return False
        else:
            self.vectorizer = self.Model(**self.init_args, dtype=np.float32)
            return True

    def fit(self, x, y=None):
        needs_fit = self._init_vectorizer()
        if needs_fit:
            self.vectorizer.fit(x, convert(y, OutputType.array))

    def forward(self, x):
        return self.vectorizer.transform(x).todense()

    def fit_transform(self, x, y=None):
        needs_fit = self._init_vectorizer()
        if needs_fit:
            return self.vectorizer.fit_transform(x, convert(y, OutputType.array))
        return self.vectorizer.transform(x)


class _TfIdfEmbeddings(_ScikitEmbeddings):
    Model = TfidfVectorizer
