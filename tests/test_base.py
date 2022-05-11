# -*- coding: utf-8 -*-

import os
import unittest

import numpy as np
import torch

from textwiser import device

docs = ['This is one document.', 'This is a second document.', 'Not the second document!']

# Make sure tests are deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BaseTest(unittest.TestCase):
    def setUp(self):
        self._reset_seed()

    def _reset_seed(self, seed=1234):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _test_fit_transform(self, tw_model, expected, atol=1e-6):
        predicted = tw_model.fit_transform(docs)
        torch.set_printoptions(precision=10)
        print(predicted)
        self.assertTrue(torch.allclose(predicted, expected.to(device), atol=atol))

    def _test_fit_before_transform(self, tw_model, expected, atol=1e-6):
        tw_model.fit(docs)
        torch.set_printoptions(precision=10)
        print(tw_model.transform(docs))
        self.assertTrue(torch.allclose(tw_model.transform(docs), expected.to(device), atol=atol))
        self.assertTrue(torch.allclose(tw_model(docs), expected.to(device), atol=atol))

    def _get_test_path(self, *names):
        cwd = os.getcwd()
        base = os.path.basename(cwd)
        path = [cwd]
        if base != 'tests':
            path.append('tests')
        path.extend(names)
        return os.path.join(*path)
