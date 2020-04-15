# -*- coding: utf-8 -*-
import os
import pickle
from tempfile import NamedTemporaryFile
import torch

from textwiser import TextWiser, Embedding, device
from tests.test_base import BaseTest, docs
from textwiser.embeddings import _TfIdfEmbeddings


class TfIdfTest(BaseTest):

    def test_fit_transform(self):
        tw = TextWiser(Embedding.TfIdf(min_df=2), dtype=torch.float32)
        expected = torch.tensor([[0.4813341796, 0.6198053956, 0.0000000000, 0.6198053956],
                                 [0.4091228545, 0.5268201828, 0.5268201828, 0.5268201828],
                                 [0.6133555174, 0.0000000000, 0.7898069024, 0.0000000000]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        self._test_fit_before_transform(tw, expected)

    def test_pretrained(self):
        tw = TextWiser(Embedding.TfIdf(pretrained=None, min_df=2), dtype=torch.float32)
        expected = torch.tensor([[0.4813341796, 0.6198053956, 0.0000000000, 0.6198053956],
                                 [0.4091228545, 0.5268201828, 0.5268201828, 0.5268201828],
                                 [0.6133555174, 0.0000000000, 0.7898069024, 0.0000000000]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)
        # Test loading from bytes
        with NamedTemporaryFile() as file:
            pickle.dump(tw._imp[0].vectorizer, file)
            file.seek(0)
            tw = TextWiser(Embedding.TfIdf(pretrained=file), dtype=torch.float32)
            predicted = tw.fit_transform(docs)
            self.assertTrue(torch.allclose(predicted, expected.to(device), atol=1e-6))
        # Test loading from file
        file_path = self._get_test_path('data', 'tfidf.pkl')
        with open(file_path, 'wb') as fp:
            pickle.dump(tw._imp[0].vectorizer, fp)
        tw = TextWiser(Embedding.TfIdf(pretrained=file_path), dtype=torch.float32)
        predicted = tw.fit_transform(docs)
        self.assertTrue(torch.allclose(predicted, expected.to(device), atol=1e-6))
        os.remove(file_path)

    def test_pretrained_error(self):
        # Not a string
        with self.assertRaises(ValueError):
            TextWiser(Embedding.TfIdf(pretrained=3), dtype=torch.float32)

        # Not a path
        with self.assertRaises(ValueError):
            TextWiser(Embedding.TfIdf(pretrained='|||||||'), dtype=torch.float32)

        # Not a path on the embedding object
        with self.assertRaises(ValueError):
            _TfIdfEmbeddings(pretrained='|||||||')._init_vectorizer()
