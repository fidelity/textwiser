# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy.sparse import csr_matrix

import textwiser.utils as utils
from tests.test_base import BaseTest, docs
from textwiser import TextWiser, Embedding, Transformation


class UtilsTest(BaseTest):

    def test_conversion(self):
        a = np.arange(10, dtype=np.float32).reshape((5, 2))
        b = csr_matrix(a)
        c = torch.from_numpy(a).to(utils.device)
        for arr in (a, b, c):
            for t in utils.OutputType:
                for dtype in (torch.float32, torch.float64) if t is utils.OutputType.tensor else (np.float32, np.float64):
                    utils.convert(arr, t, dtype)  # These should all pass

        with self.assertRaises(ValueError):
            utils.convert("a", utils.OutputType.tensor)  # This shouldn't work

    def test_set_params(self):
        # Set the arguments in container classes
        tw = TextWiser(Embedding.TfIdf(min_df=5), Transformation.NMF(n_components=30), lazy_load=True)
        tw.set_params(embedding__min_df=10, transformations__0__n_components=10)
        self.assertEqual(tw.embedding.min_df, 10)
        self.assertEqual(tw.transformations[0].n_components, 10)
        # Set the arguments in implementation
        tw = TextWiser(Embedding.Doc2Vec(vector_size=2, min_count=1, workers=1))
        tw.fit(docs)
        tw.set_params(_imp__0__seed=10)
        self.assertEqual(tw._imp[0].seed, 10)
        # Set the arguments in a schema
        schema = {
            'transform': [
                'tfidf',
                ['nmf', {'n_components': 30}]
            ]
        }
        tw = TextWiser(Embedding.Compound(schema=schema))
        tw.set_params(embedding__schema__transform__0__min_df=10, embedding__schema__transform__1__n_components=10)
        self.assertEqual(tw.embedding.schema['transform'][0][1]['min_df'], 10)
        self.assertEqual(tw.embedding.schema['transform'][1][1]['n_components'], 10)
        # Replace a part of the schema in a list
        tw.set_params(embedding__schema__transform__0='bow')
        self.assertEqual(tw.embedding.schema['transform'][0], 'bow')
        # Replace a part of the schema
        tw.set_params(embedding__schema__transform=['bow'])
        self.assertEqual(tw.embedding.schema['transform'][0], 'bow')
