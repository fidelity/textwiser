# -*- coding: utf-8 -*-
import torch

from textwiser import TextWiser, Embedding, Transformation
from tests.test_base import BaseTest


class LDATest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.LDA(n_components=2), dtype=torch.float32)
        expected = torch.tensor([[0.7724367976, 0.2275632024],
                                 [0.5895692706, 0.4104307294],
                                 [0.2381444573, 0.7618555427]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        self._reset_seed()
        self._test_fit_before_transform(tw, expected)

    def test_min_components(self):
        with self.assertRaises(ValueError):
            TextWiser(Embedding.TfIdf(min_df=2), Transformation.LDA(n_components=1), dtype=torch.float32)
