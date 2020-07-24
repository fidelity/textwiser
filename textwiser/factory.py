# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from typing import Union

from textwiser.base import BaseFeaturizer
from textwiser.embeddings import (
    _BOWEmbeddings,
    _CompoundEmbeddings,
    _Doc2VecEmbeddings,
    _RandomEmbeddings,
    _TfIdfEmbeddings,
    _USEEmbeddings,
    _WordEmbeddings,
)
from textwiser.transformations import _LDATransformation, _NMFTransformation, _PoolTransformation, _SVDTransformation, _UMAPTransformation
from textwiser.options import _ArgBase, Embedding, WordOptions, Transformation, PoolOptions, Embedding_Type, Transformation_Type
from textwiser.utils import convert, OutputType


class _Concat(BaseFeaturizer):
    def __init__(self, embeddings):
        super(_Concat, self).__init__()
        self.embeddings = nn.ModuleList(embeddings)

    def fit(self, x, y=None):
        [embedding.fit(x, y) for embedding in self.embeddings]

    def forward(self, x):
        embeds = [embedding(x) for embedding in self.embeddings]
        is_list = isinstance(embeds[0], list)  # happens for word embeddings before pooling
        types = set([OutputType.from_object(embed[0]) if is_list else OutputType.from_object(embed) for embed in embeds])
        if OutputType.tensor in types:  # need to convert everything to torch
            embeds = convert(embeds, OutputType.tensor)
            cat_fn = _Concat._tensor_concat
        elif len(types) == 2:  # both numpy and sparse
            embeds = convert(embeds, OutputType.array)
            cat_fn = _Concat._array_concat
        elif OutputType.array in types:  # only numpy
            cat_fn = _Concat._array_concat
        else:  # only sparse
            cat_fn = _Concat._sparse_concat
        if is_list:
            return [cat_fn(embed) for embed in zip(*embeds)]
        return cat_fn(embeds)

    @staticmethod
    def _tensor_concat(x):
        return torch.cat(x, -1)

    @staticmethod
    def _array_concat(x):
        return np.concatenate(x, -1)

    @staticmethod
    def _sparse_concat(x):
        return sparse.hstack(x)


class _Sequential(BaseFeaturizer, nn.Sequential):
    def fit(self, x, y=None):
        if len(self) == 1 or (len(self) == 2 and isinstance(self[1], _PoolTransformation)):
            # If there's only one model, it is an embedding, and we can just fit
            # If there's more than one, we have to do fit transforms. The exception is
            # when the only transformation is pooling, which is nonparametric and doesn't need to fit.
            self[0].fit(x, y)
        else:
            for model in self[:-1]:
                x = model.fit_transform(x, y)
            # No need for transform for the last model
            self[-1].fit(x, y)

    def fit_transform(self, x, y=None):
        for model in self:
            x = model.fit_transform(x, y)
        return x

    def forward(self, x):
        return nn.Sequential.forward(self, x)


ModelType = Union[Embedding_Type, Transformation_Type, str]


factory = {
    Embedding.Compound: _CompoundEmbeddings,
    Embedding.BOW: _BOWEmbeddings,
    Embedding.Doc2Vec: _Doc2VecEmbeddings,
    Embedding.Random: _RandomEmbeddings,
    Embedding.TfIdf: _TfIdfEmbeddings,
    Embedding.USE: _USEEmbeddings,
    Embedding.Word: _WordEmbeddings,
    Transformation.LDA: _LDATransformation,
    Transformation.NMF: _NMFTransformation,
    Transformation.Pool: _PoolTransformation,
    Transformation.SVD: _SVDTransformation,
    Transformation.UMAP: _UMAPTransformation,
}


def _get_and_init_doc_embeddings(model: ModelType, params):
    """Initializes a single document embedding object with the given params.
    Note that different models have different params; check Flair documentation
    for an up to date list. https://github.com/zalandoresearch/flair/blob/master/flair/embeddings.py
    """
    def to_word_option(model):
        if isinstance(model, str):
            if model in WordOptions.__members__:
                return WordOptions[model]
        elif isinstance(model, WordOptions):
            return model
        else:
            raise ValueError("The specified word option %s is not supported." % model)

    if 'word_option' in params:
        params['word_option'] = to_word_option(params['word_option'])

    if 'pool_option' in params and params['pool_option'] in PoolOptions.__members__:
        params['pool_option'] = PoolOptions[params['pool_option']]

    if 'inline_pool_option' in params and params['inline_pool_option'] in PoolOptions.__members__:
        params['inline_pool_option'] = PoolOptions[params['inline_pool_option']]

    # string to Embedding or Transformation conversion
    if isinstance(model, str):
        if model in WordOptions.__members__:  # is a WordOption
            model = Embedding.Word(word_option=to_word_option(model), **params)
        else:  # is a supported transformation or embedding
            model = _ArgBase.from_string(model, params)

    # model is now a _ArgBase type
    params = {**model._get_attrs(), **model.kwargs}
    # word or document level embedding
    return factory.get(model.__class__)(**params)


def get_standalone_document_embeddings(embedding):

    if isinstance(embedding, str):  # embedding type, like "tfidf"
        return _get_and_init_doc_embeddings(embedding, dict())

    if isinstance(embedding, (tuple, list)):  # type with arguments, like ('tfidf', { 'min_df': 3 }):
        return _get_and_init_doc_embeddings(embedding[0], embedding[1])

    if isinstance(embedding, _ArgBase):  # Embedding or Transformation type
        return _get_and_init_doc_embeddings(embedding, dict())


def get_document_embeddings(embeddings):

    if isinstance(embeddings, (str, tuple, list, _ArgBase)):
        return get_standalone_document_embeddings(embeddings)

    if 'transform' in embeddings:
        transforms = embeddings['transform']
        return _Sequential(get_document_embeddings(transforms[0]),
                           *[get_document_embeddings(transformation) for transformation in transforms[1:]])

    if 'concat' in embeddings:
        concats = embeddings['concat']
        return _Concat([get_document_embeddings(embedding) for embedding in concats])
