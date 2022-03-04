# -*- coding: utf-8 -*-
import torch

from textwiser import TextWiser, Embedding, Transformation, device
from tests.test_base import BaseTest, docs


class UMAPTest(BaseTest):

    def test_fit_transform(self):
        try:
            tw = TextWiser(Embedding.TfIdf(min_df=1), Transformation.UMAP(init='random', n_neighbors=2, n_components=2),
                           dtype=torch.float32)
            expected = torch.tensor([[9.6560449600, 18.2185001373],
                                     [10.3494615555, 18.9444656372],
                                     [9.7097082138, 19.6007843018]], dtype=torch.float32)
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

    def test_no_seed(self):
        try:
            tw = TextWiser(Embedding.TfIdf(min_df=1),
                           Transformation.UMAP(init='random', n_neighbors=2, n_components=2, random_state=None),
                           dtype=torch.float32)
            predicted = tw.fit_transform(docs)

            # Default result when seed is set
            expected = torch.tensor([[9.6560449600, 18.2185001373],
                                     [10.3494615555, 18.9444656372],
                                     [9.7097082138, 19.6007843018]], dtype=torch.float32)
            # The result should be different to comparing the default seed, ensuring randomness
            self.assertFalse(torch.allclose(predicted, expected.to(device), atol=1e-6))
        except ModuleNotFoundError:
            print('No UMAP found. Skipping the test. ...', end=" ", flush=True)
