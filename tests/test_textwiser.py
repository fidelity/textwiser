# -*- coding: utf-8 -*-
import numpy as np
from tempfile import NamedTemporaryFile
import torch
import torch.nn as nn
import warnings

from textwiser.options import _ArgBase

from textwiser import TextWiser, Embedding, PoolOptions, Transformation, WordOptions, device
from tests.test_base import BaseTest, docs


class TextWiserTest(BaseTest):

    def test_dtype(self):
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32)
        predicted = tw.fit_transform(docs)
        self.assertEqual(predicted.dtype, torch.float32)
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), Transformation.Pool(pool_option=PoolOptions.max), dtype=np.float32)
        predicted = tw.fit_transform(docs)
        self.assertEqual(predicted.dtype, np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), dtype=torch.float32)
            predicted = tw.fit_transform(docs)
            self.assertEqual(predicted[0].dtype, torch.float32)
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), dtype=np.float32)
            predicted = tw.fit_transform(docs)
            self.assertEqual(predicted[0].dtype, np.float32)

    def test_lazy_load(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), lazy_load=True)
            self.assertIsNone(tw._imp)
            tw.fit(docs)
            self.assertIsNotNone(tw._imp)
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                           lazy_load=True, dtype=torch.float32, is_finetuneable=True)
            self.assertIsNone(tw._imp)
            tw.fit_transform(docs)
            self.assertIsNotNone(tw._imp)

    def test_save_load(self):
        # Create a model with a downstream task
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                       [Transformation.SVD(n_components=2), Transformation.Pool(pool_option=PoolOptions.mean)], dtype=torch.float32)
        tw.fit(docs)
        model = nn.Sequential(tw, nn.Linear(2, 1)).to(device)
        # Get results of the model
        expected = model(docs)
        # Save the model to a temporary file
        with NamedTemporaryFile() as file:
            torch.save(model.state_dict(), file)  # Use string name of the file
            # Get rid of the original model
            del tw
            del model
            # Create the same model
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                           [Transformation.SVD(n_components=2), Transformation.Pool(pool_option=PoolOptions.mean)], dtype=torch.float32)
            tw.fit()
            model = nn.Sequential(tw, nn.Linear(2, 1)).to(device)
            # Load the model from file
            file.seek(0)
            model.load_state_dict(torch.load(file, map_location=device, weights_only=False))
            # Do predictions with the loaded model
            predicted = model(docs)
            self.assertTrue(torch.allclose(predicted, expected, atol=1e-6))

    def test_finetune_validation(self):
        # Nothing is fine-tuneable if dtype is numpy
        with self.assertRaises(TypeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en_turian'), dtype=np.float32, is_finetuneable=True)

        # Word2Vec is fine-tuneable
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                          dtype=torch.float32, is_finetuneable=True, lazy_load=True)
        except ValueError:
            self.fail("Word2vec is fine tuneable")

        # ELMo is not fine-tuneable, and should raise an error
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                TextWiser(Embedding.Word(word_option=WordOptions.elmo), dtype=torch.float32, is_finetuneable=True, lazy_load=True)

        # TfIdf is not fine-tuneable, and should raise an error
        with self.assertRaises(ValueError):
            TextWiser(Embedding.TfIdf(), dtype=torch.float32, is_finetuneable=True, lazy_load=True)

        # TfIdf is not fine-tuneable, but SVD is
        try:
            TextWiser(Embedding.TfIdf(), Transformation.SVD(), dtype=torch.float32, is_finetuneable=True, lazy_load=True)
        except ValueError:
            self.fail("SVD is fine tuneable")

        # LDA cannot propagate gradients, so the whole thing is not fine-tuneable
        with self.assertRaises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en'), Transformation.LDA(),
                          dtype=torch.float32, is_finetuneable=True, lazy_load=True)

        schema = {
            'concat': [
                {
                    'transform': [
                        ('word2vec', {'pretrained': 'en-turian'}),
                        ('pool', {'pool_option': 'max'})
                    ]
                },
                {
                    'transform': [
                        'tfidf',
                        ('nmf', {'n_components': 30})
                    ]
                }
            ]
        }

        # Word2Vec is fine-tuneable, therefore the whole schema is fine-tuneable
        try:
            TextWiser(Embedding.Compound(schema=schema), dtype=torch.float32, is_finetuneable=True, lazy_load=True)
        except ValueError:
            self.fail("Any fine-tuneable weights is enough for the model to be fine-tuneable")

        # TfIdf is not fine-tuneable, but SVD is
        schema = {
            'transform': [
                'tfidf',
                'svd'
            ]
        }
        try:
            TextWiser(Embedding.Compound(schema=schema), dtype=torch.float32, is_finetuneable=True, lazy_load=True)
        except ValueError:
            self.fail("SVD is fine tuneable")

    def test_forward_before_fit(self):
        """Calling `forward` before `fit` should fail"""
        with self.assertRaises(NotImplementedError):
            TextWiser(Embedding.TfIdf()).transform('document')

    def test_options_immutable(self):
        """The Embedding and Transformation options should be immutable"""
        embedding = Embedding.Doc2Vec(deterministic=False)
        with self.assertRaises(ValueError):
            embedding.deterministic = True
        self.assertFalse(embedding.deterministic)

    def test_options_from_string(self):
        self.assertIsInstance(_ArgBase.from_string('tfidf'), Embedding.TfIdf)
        with self.assertRaises(ValueError):
            _ArgBase.from_string('argbase')
