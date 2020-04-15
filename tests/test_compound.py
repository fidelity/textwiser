# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn

from textwiser import TextWiser, Embedding
from textwiser.factory import _Concat
from tests.test_base import BaseTest, docs
from textwiser.utils import device, OutputType


class CompoundTest(BaseTest):

    def test_manual_schema(self):
        schema = {
            "transform": [
                {
                    "concat": [
                        {
                            "transform": [
                                ["word", {"word_option": "word2vec", "pretrained": "en-turian"}],
                                ["pool", {"pool_option": "max"}]
                            ]
                        },
                        {
                            "transform": [
                                "tfidf",
                                ["nmf", {"n_components": 30}]
                            ]
                        }
                    ]
                },
                ["svd", {"n_components": 3}]
            ]
        }
        self._test_schema(schema)

    def test_json_schema(self):
        schema = self._get_test_path('data', 'schema.json')
        self._test_schema(schema)

    def _test_schema(self, schema):
        tw = TextWiser(Embedding.Compound(schema=schema), dtype=torch.float32)
        expected = torch.tensor([[-1.5983779430, 1.8820992708, 0.1802130789],
                                 [-1.8616007566, -0.4420076311, -0.9159148335],
                                 [-2.0401744843, -1.0712141991, 0.6945576668]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        self._reset_seed()
        self._test_fit_before_transform(tw, expected)

    def test_immutable_schema(self):
        schema = {
            "transform": [
                ["word", {"word_option": "word2vec", "pretrained": "en-turian"}],
                ["pool", {"pool_option": "max"}]
            ]
        }
        emb = Embedding.Compound(schema=schema)
        schema['transform'][1][1]['pool_option'] = 'min'
        self.assertEqual(emb.schema['transform'][1][1]['pool_option'], 'max')

    def test_no_pretrained(self):
        with self.assertRaises(ValueError):
            TextWiser(Embedding.Compound(schema='tfidf', pretrained='path'), dtype=torch.float32)

    def test_concat_types(self):
        class ArrayOut(nn.Module):
            def forward(self, _=None):
                return np.arange(10, dtype=np.float32).reshape((5, 2))

        class SparseOut(nn.Module):
            def forward(self, _=None):
                return csr_matrix(np.arange(10, dtype=np.float32).reshape((5, 2)))

        class TensorOut(nn.Module):
            def forward(self, _=None):
                return torch.from_numpy(np.arange(10, dtype=np.float32).reshape((5, 2))).to(device)

        class ListOut(nn.Module):
            def __init__(self, internal):
                super().__init__()
                self.internal = internal

            def forward(self, _=None):
                return [self.internal(), self.internal()]

        # if there is a torch tensor, everything should be converted into a torch tensor
        self.assertTrue(isinstance(_Concat([ArrayOut(), SparseOut(), TensorOut()])(docs), OutputType.tensor.value))
        self.assertTrue(isinstance(_Concat([ArrayOut(), TensorOut()])(docs), OutputType.tensor.value))
        self.assertTrue(isinstance(_Concat([SparseOut(), TensorOut()])(docs), OutputType.tensor.value))

        # if there is both numpy array and a sparse matrix, everything should be converted into a numpy array
        self.assertTrue(isinstance(_Concat([ArrayOut(), SparseOut()])(docs), OutputType.array.value))

        # if there's only sparse matrices, they can remain sparse matrices
        self.assertTrue(isinstance(_Concat([SparseOut(), SparseOut()])(docs), OutputType.sparse.value))

        # the above should also work with lists
        self.assertTrue(isinstance(_Concat([ListOut(ArrayOut()), ListOut(SparseOut()), ListOut(TensorOut())])(docs)[0], OutputType.tensor.value))
