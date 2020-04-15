# -*- coding: utf-8 -*-
import torch

from textwiser import TextWiser, Embedding, Transformation
from tests.test_base import BaseTest, docs


class SVDTest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.SVD(n_components=2), dtype=torch.float32)
        expected = torch.tensor([[-0.8526761532, 0.5070778131],
                                 [-0.9837458134, 0.0636523664],
                                 [-0.7350711226, -0.6733918786]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        self._reset_seed()
        self._test_fit_before_transform(tw, expected)

    def test_min_components(self):
        with self.assertRaises(ValueError):
            TextWiser(Embedding.TfIdf(min_df=2), Transformation.SVD(n_components=1), dtype=torch.float32)

    def test_num_components(self):
        # The natural # of components is 3.
        n_components = 2  # Restrict the # of components
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.SVD(n_components=n_components), dtype=torch.float32)
        predicted = tw.fit_transform(docs)
        self.assertEqual(predicted.shape[1], n_components)
        self._reset_seed()
        n_components = 200  # Expand the # of components
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.SVD(n_components=n_components), dtype=torch.float32)
        predicted = tw.fit_transform(docs)
        self.assertEqual(predicted.shape[1], n_components)
