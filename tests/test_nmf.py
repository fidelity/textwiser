# -*- coding: utf-8 -*-
import torch

from textwiser import TextWiser, Embedding, Transformation
from tests.test_base import BaseTest


class NMFTest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.NMF(n_components=2), dtype=torch.float32)
        expected = torch.tensor([[0.8865839243, 0.0000000000],
                                 [0.6736079454, 0.5221673250],
                                 [0.0203559380, 1.1122620106]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        self._reset_seed()
        self._test_fit_before_transform(tw, expected, atol=1e-5)

    def test_min_components(self):
        with self.assertRaises(ValueError):
            TextWiser(Embedding.TfIdf(min_df=2), Transformation.NMF(n_components=1), dtype=torch.float32)
