# -*- coding: utf-8 -*-
import torch

from textwiser import TextWiser, Embedding, Transformation
from tests.test_base import BaseTest


class UMAPTest(BaseTest):

    def test_fit_transform(self):
        try:
            tw = TextWiser(Embedding.TfIdf(min_df=1), Transformation.UMAP(init='random', n_neighbors=2, n_components=2), dtype=torch.float32)
            expected = torch.tensor([[-12.1613626480, 22.0555286407],
                                     [-11.3154125214, 22.4605998993],
                                     [-10.7626724243, 21.6793708801]], dtype=torch.float32)
            self._test_fit_transform(tw, expected)
            self._reset_seed()
            self._test_fit_before_transform(tw, expected)
        except ModuleNotFoundError:
            print('No UMAP found. Skipping the test. ...', end=" ", flush=True)

    def test_min_components(self):
        try:
            with self.assertRaises(ValueError):
                TextWiser(Embedding.TfIdf(min_df=2), Transformation.UMAP(n_components=1), dtype=torch.float32)
        except ModuleNotFoundError:
            print('No UMAP found. Skipping the test. ...', end=" ", flush=True)
