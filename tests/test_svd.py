# -*- coding: utf-8 -*-
import torch
from torch import nn, optim

from textwiser import TextWiser, Embedding, Transformation, device
from tests.test_base import BaseTest, docs


class SVDTest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.SVD(n_components=2), dtype=torch.float32)
        expected = torch.tensor([[0.8526761532, -0.5070778131],
                                 [0.9837458134, -0.0636523664],
                                 [0.7350711226, 0.6733918786]], dtype=torch.float32)
        self._test_fit_transform(tw, expected, svd=True)
        self._reset_seed()
        self._test_fit_before_transform(tw, expected, svd=True)

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

    def test_v_in_parameters(self):
        n_components = 2  # Restrict the # of components
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.SVD(n_components=n_components), dtype=torch.float32)
        tw.fit(docs)
        self.assertIn('_imp.1.V', [p[0] for p in tw.named_parameters()])

    def test_fine_tuneable(self):
        tw = TextWiser(Embedding.TfIdf(min_df=2), Transformation.SVD(n_components=2), dtype=torch.float32,
                       is_finetuneable=True)
        tw.fit(docs)
        embeddings1 = tw._imp[1].V.data.clone().detach()
        # Give a fake task to train embeddings on
        # Have a linear layer with a single output after pooling
        linear = nn.Linear(2, 1, bias=False)
        model = nn.Sequential(tw, linear).to(device).train()
        y_pred = model(docs)
        # Use ones as the target
        y_act = torch.ones_like(y_pred)
        # Optimize MSE using SGD
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        # Calculate the loss & gradients
        optimizer.zero_grad()
        loss = criterion(y_pred, y_act)
        loss.backward()
        # The embedding layer should have gradients now
        self.assertIsNotNone([p for p in tw._imp[1].named_parameters()][0][1].grad)
        # Update weights
        optimizer.step()
        # The weights should be updated if fine_tune is true, else it should be the same
        self.assertFalse(torch.allclose(embeddings1, tw._imp[1].V.data))
