# -*- coding: utf-8 -*-
import torch

from textwiser import TextWiser, Embedding
from tests.test_base import BaseTest


class BOWTest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.BOW(min_df=2), dtype=torch.float32)
        expected = torch.tensor([[1., 1., 0., 1.],
                                 [1., 1., 1., 1.],
                                 [1., 0., 1., 0.]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        self._test_fit_before_transform(tw, expected)

    def test_pretrained_error(self):
        # Not a string
        with self.assertRaises(ValueError):
            TextWiser(Embedding.BOW(pretrained=3), dtype=torch.float32)

        # Not a path
        with self.assertRaises(ValueError):
            TextWiser(Embedding.BOW(pretrained='|||||||'), dtype=torch.float32)
