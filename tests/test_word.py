# -*- coding: utf-8 -*-
import hashlib
import os

import gensim
import numpy as np
import packaging
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from textwiser import TextWiser, Embedding, PoolOptions, Transformation, WordOptions, device
from tests.test_base import BaseTest, docs
from textwiser.embeddings.word import bytepair_pretrained_decoder


class WordTest(BaseTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tested_embeddings = os.environ.get('TEST_WORD_EMBEDDINGS', None)
        tested_embeddings = set(tested_embeddings.split(',')) if tested_embeddings else set()
        tested_embeddings = {e for e in tested_embeddings}
        if 'all' in tested_embeddings:
            tested_embeddings = set(WordOptions)
        else:
            tested_embeddings = {WordOptions[e] for e in tested_embeddings}
        self.tested_embeddings = tested_embeddings

    def should_test_word_embedding(self, word_option):
        return word_option in self.tested_embeddings

    def hash(self, x):
        """Deterministic hash function for testing purposes.
        """
        return int(hashlib.sha1(x.encode('utf-8')).hexdigest(), 16) % (10 ** 8)

    def test_fit_transform(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'), dtype=torch.float32)
            predicted = torch.cat(tw.fit_transform(docs))
            expected = torch.from_numpy(np.genfromtxt(
                self._get_test_path('data', 'en_embeddings.csv'),
                dtype=np.float32)).to(device)
            self.assertTrue(torch.allclose(predicted, expected))

    def test_bytepair_pretrained_decoder(self):
        self.assertEqual(('en', 100, 10000), bytepair_pretrained_decoder('en'))
        self.assertEqual(('fr', 100, 10000), bytepair_pretrained_decoder('fr_'))
        self.assertEqual(('en', 100, 10000), bytepair_pretrained_decoder('en__'))
        self.assertEqual(('jp', 100, 5000), bytepair_pretrained_decoder('jp__5000'))
        self.assertEqual(('en', 50, 10000), bytepair_pretrained_decoder('en_50'))
        self.assertEqual(('en', 50, 10000), bytepair_pretrained_decoder('en_50_'))
        self.assertEqual(('en', 50, 5000), bytepair_pretrained_decoder('en_50_5000'))
        self.assertEqual(('en', 50, 10000), bytepair_pretrained_decoder('_50_'))
        self.assertEqual(('en', 100, 5000), bytepair_pretrained_decoder('__5000'))

    def test_vocab_matching(self):
        """Tests if the words are getting matched correctly in the vocabulary"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian')).fit()
            emb = tw._imp[0]
            # Add new words to match on vocab
            emb.vocab['tst0'] = gensim.models.keyedvectors.Vocab(index=len(emb.vocab))
            emb.vocab['test#'] = gensim.models.keyedvectors.Vocab(index=len(emb.vocab))
            self.assertEqual(emb._match_word('test'), emb._match_word('test'))  # Exact match
            self.assertEqual(emb._match_word('tesT'), emb._match_word('test'))  # Case-insensitive match
            self.assertEqual(emb._match_word('test5'), emb._match_word('test#'))  # Number-to-pound match
            self.assertEqual(emb._match_word('tst7'), emb._match_word('tst0'))  # Number-to-zero match
            self.assertEqual(emb._match_word('TST'), len(emb.vocab))  # No match, OOV

    def test_transformers_fine_tuneable(self):
        """The transformers embeddings should be fine-tuneable if is_finetuneable is True, and static if
        is_finetuneable is False."""
        hidden_sizes = {
            WordOptions.bert: 768,
            WordOptions.gpt: 768,
            WordOptions.gpt2: 1024,
            WordOptions.transformerXL: 1024,
            WordOptions.xlnet: 768,
            WordOptions.xlm: 2048,
            WordOptions.roberta: 768,
            WordOptions.distilbert: 768,
            WordOptions.ctrl: 1280,
            WordOptions.albert: 768,
            WordOptions.t5: 768,
            WordOptions.xlm_roberta: 768,
            WordOptions.bart: 768,
            WordOptions.electra: 256,
            WordOptions.dialo_gpt: 768,
            WordOptions.longformer: 768,
        }
        weights = {
            WordOptions.bert: lambda x: x._imp[0].model.embeddings.word_embeddings.weight,
            WordOptions.gpt: lambda x: x._imp[0].model.positions_embed.weight,
            WordOptions.gpt2: lambda x: x._imp[0].model.wte.weight,
            WordOptions.transformerXL: lambda x: x._imp[0].model.word_emb.emb_layers[0].weight,
            WordOptions.xlnet: lambda x: x._imp[0].model.word_embedding.weight,
            WordOptions.xlm: lambda x: x._imp[0].model.position_embeddings.weight,
            WordOptions.roberta: lambda x: x._imp[0].model.embeddings.word_embeddings.weight,
            WordOptions.distilbert: lambda x: x._imp[0].model.embeddings.word_embeddings.weight,
            WordOptions.ctrl: lambda x: x._imp[0].model.w.weight,
            WordOptions.albert: lambda x: x._imp[0].model.embeddings.word_embeddings.weight,
            WordOptions.t5: lambda x: x._imp[0].model.shared.weight,
            WordOptions.xlm_roberta: lambda x: x._imp[0].model.embeddings.word_embeddings.weight,
            WordOptions.bart: lambda x: x._imp[0].model.shared.weight,
            WordOptions.electra: lambda x: x._imp[0].model.embeddings.word_embeddings.weight,
            WordOptions.dialo_gpt: lambda x: x._imp[0].model.wte.weight,
            WordOptions.longformer: lambda x: x._imp[0].model.embeddings.word_embeddings.weight,
        }
        print()
        for o in WordOptions:
            if not self.should_test_word_embedding(o):
                print('`TEST_WORD_EMBEDDINGS` environmental variable does not contain `{}` or `all`. Skipping the test. ...'.format(o.name))
                continue
            if not o.is_from_transformers():
                continue
            if o == WordOptions.transformerXL and packaging.version.parse(torch.__version__) < packaging.version.parse('1.2'):
                print("%s only works with PyTorch >= 1.2. Skipping the test. ..." % o, flush=True)
                continue
            print('Testing {}...'.format(o), end=" ", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tw = TextWiser(Embedding.Word(word_option=o), Transformation.Pool(pool_option=PoolOptions.max),
                               dtype=torch.float32, is_finetuneable=True)
                self._test_fine_tuning(tw, weights[o], dim=hidden_sizes[o], fine_tune=True)
                print('ok', flush=True)

    def test_word2vec_fit(self):
        """The word2vec embeddings should be trainable from scratch."""
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained=None, seed=1234, vector_size=2,
                                      min_count=1, workers=1, sample=0, negative=5, hashfxn=self.hash),
                       Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32)
        expected = torch.tensor([[0.4905824959, 0.4688255489],
                                 [0.4905824959, 0.4849084020],
                                 [0.4857406616, 0.4849084020]], dtype=torch.float32)
        self._test_fit_transform(tw, expected)

    def test_tokenizer_validation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # shouldn't raise an error
            try:
                TextWiser(Embedding.Word(word_option=WordOptions.word2vec, tokenizer=lambda doc: doc.lower().split()))
            except TypeError:
                self.fail("This tokenizer should pass the validation.")

            # should raise the first error
            with self.assertRaises(ValueError):
                TextWiser(Embedding.Word(word_option=WordOptions.bert, tokenizer=lambda doc: doc.lower().split()))

            # should raise the second error
            with self.assertRaises(TypeError):
                TextWiser(Embedding.Word(word_option=WordOptions.word2vec, tokenizer=lambda doc: doc.lower()))

            # should raise the third error
            with self.assertRaises(TypeError):
                TextWiser(Embedding.Word(word_option=WordOptions.word2vec, tokenizer=lambda doc: [1]))

    def _test_fine_tuning(self, tw, get_weight, dim=50, fine_tune=True):
        tw.fit()
        embeddings1 = get_weight(tw).clone().detach()
        # Give a fake task to train embeddings on
        # Have a linear layer with a single output after pooling
        linear = nn.Linear(dim, 1, bias=False)
        model = nn.Sequential(tw, linear).to(device).train()
        y_pred = model(docs)
        # Use ones as the target
        y_act = torch.ones_like(y_pred)
        # Optimize MSE using SGD
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-4)
        # Calculate the loss & gradients
        optimizer.zero_grad()
        loss = criterion(y_pred, y_act)
        loss.backward()
        # The embedding layer should have gradients now if fine_tune is true, else it should be none
        self.assertIsNotNone(get_weight(tw).grad) if fine_tune else self.assertIsNone(get_weight(tw).grad)
        # Update weights
        optimizer.step()
        # The weights should be updated if fine_tune is true, else it should be the same
        self.assertTrue(torch.allclose(embeddings1, get_weight(tw)) != fine_tune)

    def test_bytepair_fine_tuneable(self):
        """The bytepair embeddings should be fine-tuneable if is_finetuneable is True, and static if
        is_finetuneable is False."""
        tw = TextWiser(Embedding.Word(word_option=WordOptions.bytepair, pretrained='en_25_1000'),
                       Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32, is_finetuneable=True)
        self._test_fine_tuning(tw, lambda x: x._imp[0].model.weight, dim=25, fine_tune=True)
        tw = TextWiser(Embedding.Word(word_option=WordOptions.bytepair, pretrained='en_25_1000'),
                       Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32, is_finetuneable=False)
        self._test_fine_tuning(tw, lambda x: x._imp[0].model.weight, dim=25, fine_tune=False)

    def test_word2vec_fine_tuneable(self):
        """The word2vec embeddings should be fine-tuneable if is_finetuneable is True, and static if
        is_finetuneable is False."""
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                       Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32, is_finetuneable=True)
        self._test_fine_tuning(tw, lambda x: x._imp[0].model.weight, dim=50, fine_tune=True)
        tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                       Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32, is_finetuneable=False)
        self._test_fine_tuning(tw, lambda x: x._imp[0].model.weight, dim=50, fine_tune=False)

    def test_flair_fine_tuneable(self):
        """The Flair embeddings should be fine-tuneable if is_finetuneable is True, and static if
        is_finetuneable is False."""
        tw = TextWiser(Embedding.Word(word_option=WordOptions.flair, pretrained='news-forward-fast'),
                       Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32, is_finetuneable=True)
        self._test_fine_tuning(tw, lambda x: x._imp[0].model.lm.encoder.weight, dim=1024, fine_tune=True)
        tw = TextWiser(Embedding.Word(word_option=WordOptions.flair, pretrained='news-forward-fast'),
                       Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32, is_finetuneable=False)
        self._test_fine_tuning(tw, lambda x: x._imp[0].model.lm.encoder.weight, dim=1024, fine_tune=False)

    def test_layers(self):
        """The layers parameter should give different hidden layers in transformer models"""
        # Layers shouldn't do anything in word2vec
        default_vecs = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                                 Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32).fit_transform(docs)
        last_layer_vecs = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian', layers=-1),
                                    Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32).fit_transform(docs)
        random_layer_vecs = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian', layers=10000),
                                      Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32).fit_transform(docs)
        self.assertTrue(torch.equal(default_vecs, last_layer_vecs))
        self.assertTrue(torch.equal(default_vecs, random_layer_vecs))
        # Layers should fail if given an incorrect input type
        with self.assertRaises(ValueError):
            TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian', layers='textwiser'))
        # Layers should work for transformer models
        if self.should_test_word_embedding(WordOptions.distilbert):
            default_vecs = TextWiser(Embedding.Word(word_option=WordOptions.distilbert),
                                     Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32).fit_transform(docs)
            last_layer_vecs = TextWiser(Embedding.Word(word_option=WordOptions.distilbert, layers=-1),
                                        Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32).fit_transform(docs)
            penultimate_layer_vecs = TextWiser(Embedding.Word(word_option=WordOptions.distilbert, layers=-2),
                                               Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32).fit_transform(docs)
            last_two_layer_vecs = TextWiser(Embedding.Word(word_option=WordOptions.distilbert, layers=[-1, -2]),
                                            Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32).fit_transform(docs)
            self.assertTrue(torch.equal(default_vecs, last_layer_vecs))
            self.assertTrue(torch.equal(last_layer_vecs, last_two_layer_vecs[:, :768]))
            self.assertTrue(torch.equal(penultimate_layer_vecs, last_two_layer_vecs[:, 768:]))
        else:
            print('`TEST_WORD_EMBEDDINGS` environmental variable does not contain `{}` or `all`. Skipping the test. ...'.format(
                    WordOptions.distilbert.name), end=" ", flush=True)

    def test_embeddings_from_path(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tw = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained=self._get_test_path('data', 'dummy.kv')), dtype=torch.float32)
            predicted = torch.cat(tw.fit_transform(docs))
            expected = torch.tensor([[-0.0347005501, 0.0904469639, -0.0338280797],
                                     [-0.0547435991, 0.0380357727, 0.0247183330],
                                     [0.1166398972, 0.0895400494, -0.0631256625],
                                     [0.0000000000, 0.0000000000, 0.0000000000],
                                     [-0.0347005501, 0.0904469639, -0.0338280797],
                                     [-0.0547435991, 0.0380357727, 0.0247183330],
                                     [0.0184323676, 0.0117521724, -0.1661947370],
                                     [-0.0584235601, 0.0191440862, 0.1507517248],
                                     [0.0000000000, 0.0000000000, 0.0000000000],
                                     [0.0267334618, -0.0343993008, -0.1664674431],
                                     [-0.0790654048, 0.1425552219, 0.0537007749],
                                     [-0.0584235601, 0.0191440862, 0.1507517248],
                                     [0.0000000000, 0.0000000000, 0.0000000000]], dtype=torch.float32)
            self.assertTrue(torch.allclose(predicted, expected.to(device)))

    def test_all_word_embeddings(self):
        long_doc = ['  '.join(['unfathomable'] * 1025)]
        textwiser_params = {
            WordOptions.char: {'is_finetuneable': True},
        }
        embedding_params = {
            WordOptions.elmo: {'output_layer': "elmo"},
            WordOptions.xlnet: {'pretrained': 'xlnet-base-cased'},
        }
        print()
        for o in WordOptions:
            if not self.should_test_word_embedding(o):
                print('`TEST_WORD_EMBEDDINGS` environmental variable does not contain `{}` or `all`. Skipping the test. ...'.format(o.name))
                continue
            if o == WordOptions.elmo:
                try:
                    import tensorflow
                    import tensorflow_hub
                except ModuleNotFoundError:
                    print("%s only works with Tensorflow and TensorflowHub installed. "
                          "Skipping the test. ..." % o, flush=True)
                    continue
            if o == WordOptions.transformerXL and packaging.version.parse(torch.__version__) < packaging.version.parse('1.2'):
                print("%s only works with PyTorch >= 1.2. Skipping the test. ..." % o, flush=True)
                continue
            print('Testing {}...'.format(o), end=" ", flush=True)
            tw_params = textwiser_params.get(o, dict())
            emb_params = embedding_params.get(o, dict())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tw = TextWiser(Embedding.Word(word_option=o, **emb_params),
                               Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32, **tw_params).fit()
                self.assertTrue(isinstance(tw.transform(docs), torch.Tensor))
                self.assertTrue(isinstance(tw.transform(long_doc), torch.Tensor))
                print('ok', flush=True)

    def test_inline_pool(self):
        # Test that pooling in Embedding.Word gives the same result as using pool transformation
        tw1 = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian'),
                        Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32)
        tw2 = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian',
                                       inline_pool_option=PoolOptions.max), dtype=torch.float32)
        target = tw1.fit_transform(docs)
        self.assertTrue(torch.allclose(target, tw2.fit_transform(docs)))

        # Test that inline pooling can be done through the schema
        tw3 = TextWiser(Embedding.Compound(schema=["word", {"word_option": "word2vec", "pretrained": "en-turian", "inline_pool_option": "max"}]), dtype=torch.float32)
        self.assertTrue(torch.allclose(target, tw3.fit_transform(docs)))

        # Test that double pooling raises an error
        with self.assertRaises(ValueError):
            TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained='en-turian',
                                     inline_pool_option=PoolOptions.max),
                      Transformation.Pool(pool_option=PoolOptions.max), dtype=torch.float32)
