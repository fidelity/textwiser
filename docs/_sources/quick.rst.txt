.. _quick:

Quick Start
===========

.. code-block:: python

    # Conceptually, TextWiser is composed of an Embedding, potentially with a pretrained model,
    # that can be chained into zero or more Transformations
    from textwiser import TextWiser, Embedding, Transformation, WordOptions, PoolOptions

    # Data
    documents = ["Some document", "More documents. Including multi-sentence documents."]

    # Model: TFIDF `min_df` parameter gets passed to sklearn automatically
    emb = TextWiser(Embedding.TfIdf(min_df=1))

    # Model: TFIDF followed with an NMF + SVD
    emb = TextWiser(Embedding.TfIdf(min_df=1), [Transformation.NMF(n_components=30), Transformation.SVD(n_components=10)])

    # Model: Word2Vec with no pretraining that learns from the input data
    emb = TextWiser(Embedding.Word(word_option=WordOptions.word2vec, pretrained=None), Transformation.Pool(pool_option=PoolOptions.min))

    # Model: BERT with the pretrained bert-base-uncased embedding
    emb = TextWiser(Embedding.Word(word_option=WordOptions.bert), Transformation.Pool(pool_option=PoolOptions.first))

    # Features
    vecs = emb.fit_transform(documents)
