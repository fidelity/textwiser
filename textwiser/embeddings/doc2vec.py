# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0
import warnings

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import os
import pickle

from textwiser.base import BaseFeaturizer
from textwiser.utils import split_tokenizer, Constants


class _Doc2VecEmbeddings(BaseFeaturizer):
    def __init__(self, pretrained=None, tokenizer=None, deterministic=False, **kwargs):
        super(_Doc2VecEmbeddings, self).__init__()
        self.pretrained = pretrained
        self.model = None
        self.init_args = kwargs
        self.tokenizer = tokenizer if tokenizer else split_tokenizer
        self.deterministic = deterministic
        self.seed = kwargs['seed'] if 'seed' in kwargs else 123456

    def fit(self, x, y=None):
        def fit_model():
            if 'PYTHONHASHSEED' not in os.environ:
                # Since gensim 4.0, Doc2Vec is only reproducible when ``PYTHONHASHSEED`` is set. So we can give a
                # warning to the users if they're not using it when training a model.
                warnings.warn(
                    "The ``PYTHONHASHSEED`` environmental variable isn't set. If you want to get a reproducible"
                    "Doc2Vec model, please set ``PYTHONHASHSEED`` and run your training script again.")
            documents = [TaggedDocument(self.tokenizer(doc), [i]) for i, doc in enumerate(x)]
            self.model = Doc2Vec(documents, **self.init_args)

        if isinstance(self.pretrained, str):
            if self.pretrained is Constants.default_model:
                fit_model()
            elif os.path.exists(self.pretrained):
                with open(self.pretrained, 'rb') as fp:
                    self.model = pickle.load(fp)
            else:
                raise ValueError("Cannot find the specified path for pretrained model.")
        elif hasattr(self.pretrained, 'read'):  # File-like
            self.model = pickle.load(self.pretrained)
        else:
            fit_model()

    def _infer(self, doc):
        if self.deterministic:
            self.model.random.seed(self.seed)
        return self.model.infer_vector(self.tokenizer(doc))

    def forward(self, x):
        return np.stack([self._infer(doc) for doc in x])
