# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import os

import flair
import transformers

from textwiser.embeddings.bow import _BOWEmbeddings
from textwiser.embeddings.compound import _CompoundEmbeddings
from textwiser.embeddings.doc2vec import _Doc2VecEmbeddings
from textwiser.embeddings.random import _RandomEmbeddings
from textwiser.embeddings.tfidf import _TfIdfEmbeddings
from textwiser.embeddings.use import _USEEmbeddings
from textwiser.embeddings.word import _WordEmbeddings


def collapse_user(path: str) -> str:
    """Collapse given path to exclude home directory."""
    home_dir = os.path.expanduser("~")
    if path.startswith(home_dir):
        path = "~" + path[len(home_dir):]
    return path


def set_cache_locations():
    """Set model download locations for each library."""
    flair.cache_root = collapse_user(str(flair.cache_root))
    transformers.TRANSFORMERS_CACHE = collapse_user(str(transformers.TRANSFORMERS_CACHE))
    os.environ["TFHUB_CACHE_DIR"] = collapse_user(str(os.environ.get("TFHUB_CACHE_DIR", "~/.cache/tfhub_modules")))


set_cache_locations()
