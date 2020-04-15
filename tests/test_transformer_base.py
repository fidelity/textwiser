# -*- coding: utf-8 -*-
import torch
import warnings

from textwiser import TextWiser, Embedding, Transformation, WordOptions, device
from tests.test_base import BaseTest, docs


class BaseTransformerTest(BaseTest):

    def test_list_handling(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), Transformation.SVD(n_components=2), dtype=torch.float32)
            predicted = tw.fit_transform(docs)
            expected = [
                torch.tensor([[-0.9719871283, 0.0947150663],
                              [-0.3805825114, -1.0427029133],
                              [-0.6929296255, 0.1793890595],
                              [0.0000000000, 0.0000000000]], dtype=torch.float32),
                torch.tensor([[-0.9719871283, 0.0947150663],
                              [-0.3805825114, -1.0427029133],
                              [-0.7170552015, 0.0105144158],
                              [-0.9385635853, 0.6596723199],
                              [0.0000000000, 0.0000000000]], dtype=torch.float32),
                torch.tensor([[-0.8687936068, -0.9333068132],
                              [-0.6859120131, 0.0732812732],
                              [-0.9385635853, 0.6596723199],
                              [0.0000000000, 0.0000000000]], dtype=torch.float32)
            ]
            for p, e in zip(predicted, expected):
                self.assertTrue(torch.allclose(p, e.to(device), atol=1e-6))
