# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

from sklearn.feature_extraction.text import CountVectorizer

from textwiser.embeddings.tfidf import _ScikitEmbeddings


class _BOWEmbeddings(_ScikitEmbeddings):
    Model = CountVectorizer
