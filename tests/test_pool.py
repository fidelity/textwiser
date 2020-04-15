# -*- coding: utf-8 -*-
import warnings

import numpy as np
import torch

from textwiser import TextWiser, Embedding, PoolOptions, Transformation, WordOptions, device
from tests.test_base import BaseTest, docs
from textwiser.transformations import _PoolTransformation


class PoolTest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32)
        expected = torch.from_numpy(np.genfromtxt(
            self._get_test_path('data', 'pooled_embeddings.csv'),
            dtype=np.float32))
        self._test_fit_transform(tw, expected)
        self._test_fit_before_transform(tw, expected)

    def _test_index(self, pool_option):
        index = 0 if pool_option == PoolOptions.first else -1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                           dtype=torch.float32)
            expected = tw.fit_transform(docs[0])[0][index].view(1, -1)
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                       Transformation.Pool(pool_option=pool_option), dtype=torch.float32)
        pooled = tw.fit_transform(docs[0])
        self.assertTrue(torch.allclose(expected.to(device), pooled.to(device)))

    def test_options(self):
        dim = 5
        shape = (2, dim)
        vecs = [torch.stack((torch.zeros(dim), torch.ones(dim)), dim=0)]
        # Test max
        pooler = _PoolTransformation(PoolOptions.max)
        self.assertTrue(torch.allclose(pooler._forward(vecs)[0], torch.ones(dim)))
        # Test mean
        pooler = _PoolTransformation(PoolOptions.mean)
        self.assertTrue(torch.allclose(pooler._forward(vecs)[0], torch.ones(dim) * 0.5))
        # Test min
        pooler = _PoolTransformation(PoolOptions.min)
        self.assertTrue(torch.allclose(pooler._forward(vecs)[0], torch.zeros(dim)))
        # Test first
        self._test_index(PoolOptions.first)
        # Test last
        self._test_index(PoolOptions.last)
