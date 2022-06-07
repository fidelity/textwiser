# -*- coding: utf-8 -*-
import hashlib
import logging
import os
import pickle
from tempfile import NamedTemporaryFile
import torch

from textwiser import TextWiser, Embedding, device
from tests.test_base import BaseTest, docs
from textwiser.embeddings import _Doc2VecEmbeddings

logger = logging.getLogger('gensim')
logger.setLevel(logging.ERROR)


def det_hash(x):
    """Deterministic hash function for testing purposes.
    """
    return int(hashlib.sha1(x.encode('utf-8')).hexdigest(), 16) % (10 ** 8)


class Doc2VecTest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.Doc2Vec(seed=1234, vector_size=2, min_count=1, workers=1, sample=0, negative=0,
                                         hashfxn=det_hash), dtype=torch.float32)
        expected = torch.tensor([[0.1922276616,  0.2392318249],
                                 [-0.0607281923,  0.0164829195],
                                 [0.0748119354, -0.0934505463]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)

    def test_deterministic_transform(self):
        """Specifying the `deterministic` option should make Doc2Vec transformation deterministic.

        By default, running inference with doc2vec is not deterministic in gensim.
        This test makes sure we can get a deterministic result when necessary.
        """
        tw = TextWiser(Embedding.Doc2Vec(deterministic=True, seed=1234, vector_size=2, min_count=1, workers=1, sample=0,
                                         negative=0, hashfxn=det_hash), dtype=torch.float32)
        expected = torch.tensor([[0.1922276616,  0.2392318249],
                                 [-0.0607281923,  0.0164829195],
                                 [0.0748119354, -0.0934505463]], dtype=torch.float32)
        self._test_fit_before_transform(tw, expected)
        tw = TextWiser(Embedding.Doc2Vec(pretrained=None, deterministic=True, seed=1234, vector_size=2, min_count=1,
                                         workers=1, sample=0, negative=0, hashfxn=det_hash), dtype=torch.float32)
        self._test_fit_before_transform(tw, expected)


    def test_tokenizer_validation(self):
        # shouldn't raise an error
        try:
            TextWiser(Embedding.Doc2Vec(tokenizer=lambda doc: doc.lower().split()))
        except TypeError:
            self.fail("This tokenizer should pass the validation.")

        # should raise the first error
        with self.assertRaises(TypeError):
            TextWiser(Embedding.Doc2Vec(tokenizer=lambda doc: doc.lower()))

        # should raise the second error
        with self.assertRaises(TypeError):
            TextWiser(Embedding.Doc2Vec(tokenizer=lambda doc: [1]))

    def test_pretrained(self):
        tw = TextWiser(Embedding.Doc2Vec(deterministic=True, seed=1234, vector_size=2, min_count=1, workers=1, sample=0, negative=0, hashfxn=det_hash), dtype=torch.float32)
        expected = torch.tensor([[0.1922276616,  0.2392318249],
                                 [-0.0607281923,  0.0164829195],
                                 [0.0748119354, -0.0934505463]], dtype=torch.float32)
        self._test_fit_before_transform(tw, expected)
        # Test loading from bytes
        with NamedTemporaryFile() as file:
            pickle.dump(tw._imp[0].model, file)
            file.seek(0)
            tw = TextWiser(Embedding.Doc2Vec(pretrained=file, deterministic=True, seed=1234), dtype=torch.float32)
            predicted = tw.fit_transform(docs)
            self.assertTrue(torch.allclose(predicted, expected.to(device), atol=1e-6))
        # Test loading from file
        file_path = self._get_test_path('data', 'doc2vec.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(tw._imp[0].model, fp)
        tw = TextWiser(Embedding.Doc2Vec(pretrained=file_path, deterministic=True, seed=1234), dtype=torch.float32)
        predicted = tw.fit_transform(docs)
        self.assertTrue(torch.allclose(predicted, expected.to(device), atol=1e-6))
        os.remove(file_path)

    def test_pretrained_error(self):
        # Not a string
        with self.assertRaises(ValueError):
            TextWiser(Embedding.Doc2Vec(pretrained=3), dtype=torch.float32)

        # Not a path
        with self.assertRaises(ValueError):
            TextWiser(Embedding.Doc2Vec(pretrained='|||||||'), dtype=torch.float32)

        # Not a path on the embedding object
        with self.assertRaises(ValueError):
            _Doc2VecEmbeddings(pretrained='|||||||').fit([])
