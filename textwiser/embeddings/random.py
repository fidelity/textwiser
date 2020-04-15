# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import numpy as np
from scipy.stats import bernoulli, beta, expon, lognorm, norm, uniform

from textwiser.options import WordOptions
from textwiser.embeddings.compound import _CompoundEmbeddings

embeddings = ['transform', 'concat', 'bow', 'doc2vec', 'tfidf', 'use', 'word']
transformations = ['lda', 'nmf', 'svd', 'umap']  # handle pool as a special case


def random_n_components(loc=2, a=2, b=3, scale=100, size=1, **kwargs):
    return int(np.floor(beta.rvs(loc=loc, a=a, b=b, scale=scale, size=size)).item())


def random_transform(child_embedding=None, concat_scale=0.6, transform_scale=0.6, **kwargs):
    child_embedding = child_embedding if child_embedding else random_schema(concat_scale=concat_scale, transform_scale=transform_scale)
    n_transforms = int(np.floor(expon.rvs(loc=1, scale=transform_scale, size=1)).item())
    transforms = [random_transformation() for _ in range(n_transforms)]
    if child_embedding == 'word':
        child_embedding = random_word()
        # TODO this restricts the search space: you can't first concat two embeddings, do SVD, and then pool them
        # How different is that to SVD + pool + concat?
        weakest_link = np.random.randint(n_transforms)
        transforms[weakest_link] = random_pool()
    return {'transform': [child_embedding] + transforms}


def random_concat(concat_scale=0.6, transform_scale=0.6, **kwargs):
    n_embs = int(np.floor(expon.rvs(loc=2, scale=concat_scale, size=1)).item())
    return {'concat': [random_schema(concat_scale=concat_scale, transform_scale=transform_scale) for _ in range(n_embs)]}


def random_bow(df_scale=0.1, **kwargs):
    return ('bow', {
        'max_df': 1 - expon.rvs(loc=0, scale=df_scale, size=1).item(),  # max % of times a term can be found
        'min_df': expon.rvs(loc=0, scale=df_scale, size=1).item(),  # min % of times a term can be found
        # 'max_features':  # not sure how to randomly pick this
        'binary': bernoulli.rvs(0.2, size=1).item(),  # whether to make bow binary
    })


def random_doc2vec(min_count_scale=10, **kwargs):
    return ('doc2vec', {
        'vector_size': int(np.floor(beta.rvs(loc=2, a=2, b=3, scale=100, size=1)).item()),  # Dimensionality of the feature vectors.
        'window': max(1, int(norm.rvs(loc=5, scale=1, size=1).item())),  # the maximum distance between the current and predicted word within a sentence.
        'min_count': int(np.floor(beta.rvs(loc=2, a=2, b=2.5, scale=min_count_scale, size=1)).item()),  # Ignores all words with total frequency lower than this.
        'max_vocab_size': int(np.floor(np.exp(lognorm.rvs(loc=9, s=0.1, scale=4, size=1)))),  # Limits the vocabulary; if there are more unique words than this, then prune the infrequent ones
        'sample': uniform.rvs(loc=0, scale=1e-5, size=1).item(),  # The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
        'dm_mean': bernoulli.rvs(0.4, size=1).item(),  # whether to use the sum of the context word vectors instead of the mean
        'dm_concat': bernoulli.rvs(0.05, size=1).item(),  # whether to use concatenation of context vectors rather than sum/average
    })


def random_tfidf(df_scale=0.1, **kwargs):
    return ('tfidf', {
        'max_df': 1 - expon.rvs(loc=0, scale=df_scale, size=1).item(),  # max # of times a term can be found
        'min_df': expon.rvs(loc=0, scale=df_scale, size=1).item(),  # min # of times a term can be found
        # 'max_features':  # not sure how to randomly pick this
        'binary': bernoulli.rvs(0.2, size=1).item(),  # whether to make bow binary
        'norm': np.random.choice(['l2', 'l1', None], p=[0.8, 0.15, 0.05]),  # how to normalize the vectors
    })


def random_word(**kwargs):
    return np.random.choice(list([o.value for o in WordOptions]))


def random_lda(**kwargs):
    return ('lda', {
        'n_components': random_n_components(**kwargs),  # number of dimensions
    })


def random_nmf(**kwargs):
    return ('nmf', {
        'n_components': random_n_components(**kwargs),  # number of dimensions
        'alpha': expon.rvs(loc=0, scale=0.1, size=1).item(),  # regularization strength
        'l1_ratio': uniform.rvs(size=1).item(),  # ratio of L1 to L2
    })


def random_svd(**kwargs):
    return ('svd', {
        'n_components': random_n_components(**kwargs),  # number of dimensions
    })


def random_umap(**kwargs):
    return ('umap', {
        'n_components': random_n_components(**kwargs),  # number of dimensions
    })


def random_pool(**kwargs):
    return ('pool', {
        'pool_option': np.random.choice(['min', 'mean', 'max', 'first', 'last']),
    })


emb_map = {
    'transform': random_transform,
    'concat': random_concat,
    'bow': random_bow,
    'doc2vec': random_doc2vec,
    'tfidf': random_tfidf,
    'lda': random_lda,
    'nmf': random_nmf,
    'svd': random_svd,
    'umap': random_umap,
}


def random_embedding(concat_scale=0.6, transform_scale=0.6, available_embeddings=None):
    emb = np.random.choice(available_embeddings if available_embeddings else embeddings)
    if emb in emb_map:
        return emb_map[emb](concat_scale=concat_scale, transform_scale=transform_scale)
    return emb


def random_transformation(available_transformations=None):
    tfm = np.random.choice(available_transformations if available_transformations else transformations)
    if tfm in emb_map:
        return emb_map[tfm]()
    return tfm


def random_schema(preconfigured_embedding=None, concat_scale=0.6, transform_scale=0.6, available_embeddings=None):
    emb = preconfigured_embedding if preconfigured_embedding else random_embedding(concat_scale=concat_scale, transform_scale=transform_scale, available_embeddings=available_embeddings)
    if emb == 'word':
        return random_transform(child_embedding='word', concat_scale=concat_scale, transform_scale=transform_scale)
    return emb


class _RandomEmbeddings(_CompoundEmbeddings):
    def __init__(self, pretrained=None, **kwargs):
        super(_RandomEmbeddings, self).__init__(random_schema(**kwargs))
