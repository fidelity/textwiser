# -*- coding: utf-8 -*-
import torch
import textwiser.embeddings.random as rnd

from textwiser import TextWiser, Embedding
from tests.test_base import BaseTest


class RandomTest(BaseTest):

    def _test_output_type(self, res, str_name):
        self.assertIsInstance(res, tuple)
        self.assertIsInstance(res[1], dict)
        self.assertEqual(res[0], str_name)

    def test_fit_transform(self):
        self._reset_seed(seed=12345)
        tw = TextWiser(Embedding.Random(), dtype=torch.float32)
        expected = torch.tensor([[1., 0., 1., 0., 0., 1.],
                                 [1., 0., 0., 1., 0., 1.],
                                 [0., 1., 0., 1., 1., 0.]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        self._test_fit_before_transform(tw, expected)

    def test_random_transform(self):
        res = rnd.random_transform()
        self.assertIsInstance(res, dict)
        self.assertIn('transform', res)
        res = res['transform']
        self.assertIsInstance(res, list)

        res = rnd.random_transform(child_embedding='word')
        transforms = [t[0] if isinstance(t, tuple) else t for t in res['transform']]
        self.assertIn('pool', transforms)

    def test_random_concat(self):
        res = rnd.random_concat()
        self.assertIsInstance(res, dict)
        self.assertIn('concat', res)
        res = res['concat']
        self.assertIsInstance(res, list)

    def test_random_tfidf(self):
        res = rnd.random_tfidf()
        self._test_output_type(res, 'tfidf')
        self.assertEqual(res[1]['binary'], 0)
        self.assertEqual(res[1]['norm'], 'l2')

    def test_random_nmf(self):
        res = rnd.random_nmf()
        self._test_output_type(res, 'nmf')
        self.assertEqual(res[1]['n_components'], 69)
        self.assertAlmostEqual(res[1]['alpha'], 0.1514017775578926)
        self.assertAlmostEqual(res[1]['l1_ratio'], 0.2725926052826416)

    def test_random_svd(self):
        res = rnd.random_svd()
        self._test_output_type(res, 'svd')
        self.assertEqual(res[1], {'n_components': 69})

    def test_random_umap(self):
        res = rnd.random_umap()
        self._test_output_type(res, 'umap')
        self.assertEqual(res[1], {'n_components': 69})

    def test_random_pool(self):
        res = rnd.random_pool()
        self._test_output_type(res, 'pool')
        self.assertEqual(res[1], {'pool_option': 'first'})

    def test_random_transformation(self):
        res = rnd.random_transformation()
        self.assertEqual(res[0], 'umap')
        res = rnd.random_transformation(available_transformations=['svd'])
        self.assertEqual(res[0], 'svd')
