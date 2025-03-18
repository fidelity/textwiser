# -*- coding: utf-8 -*-
from tempfile import NamedTemporaryFile

import numpy as np
import os
import torch
from torch import nn

from textwiser import TextWiser, Embedding, device
from tests.test_base import BaseTest, docs


class USETest(BaseTest):

    def test_fit_transform(self):
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut tensorflow up during testing
            tw = TextWiser(Embedding.USE(), dtype=torch.float32)
            expected = torch.from_numpy(np.genfromtxt(
                self._get_test_path('data', 'use_embeddings.csv'),
                dtype=np.float32))
            self._test_fit_transform(tw, expected)
            self._test_fit_before_transform(tw, expected)
        except ModuleNotFoundError:
            print('No Tensorflow found. Skipping the test. ...', end=" ", flush=True)

    def test_pretrained_error(self):
        # Not a pretrained model
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut tensorflow up during testing
            with self.assertRaises(ValueError):
                TextWiser(Embedding.USE(pretrained=None), dtype=torch.float32)
        except ModuleNotFoundError:
            print('No Tensorflow found. Skipping the test. ...', end=" ", flush=True)

    def test_save_load(self):
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut tensorflow up during testing
            # Create a model with a downstream task
            tw = TextWiser(Embedding.USE(), dtype=torch.float32).fit(docs)
            model = nn.Sequential(tw, nn.Linear(512, 1)).to(device)
            # Get results of the model
            expected = model(docs)
            # Save the model to a temporary file
            with NamedTemporaryFile() as file:
                state_dict = model.state_dict()
                self.assertNotIn('0._imp.0.use', state_dict)
                torch.save(state_dict, file)  # Use string name of the file
                # Get rid of the original model
                del tw
                del model
                # Create the same model
                tw = TextWiser(Embedding.USE(), dtype=torch.float32)
                tw.fit()
                model = nn.Sequential(tw, nn.Linear(512, 1)).to(device)
                # Load the model from file
                file.seek(0)
                model.load_state_dict(torch.load(file, map_location=device, weights_only=False))
                # Do predictions with the loaded model
                predicted = model(docs)
                self.assertTrue(torch.allclose(predicted, expected, atol=1e-6))
        except ModuleNotFoundError:
            print('No Tensorflow found. Skipping the test. ...', end=" ", flush=True)

    def test_use_versions(self):
        """Tests if the previous versions of USE are useable"""
        try:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut tensorflow up during testing
            TextWiser(Embedding.USE(pretrained='https://tfhub.dev/google/universal-sentence-encoder-large/5'),
                      dtype=torch.float32).fit_transform(docs)
            TextWiser(Embedding.USE(pretrained='https://tfhub.dev/google/universal-sentence-encoder-large/4'),
                      dtype=torch.float32).fit_transform(docs)
            TextWiser(Embedding.USE(pretrained='https://tfhub.dev/google/universal-sentence-encoder-large/3'),
                      dtype=torch.float32).fit_transform(docs)
        except ModuleNotFoundError:
            print('No Tensorflow found. Skipping the test. ...', end=" ", flush=True)
