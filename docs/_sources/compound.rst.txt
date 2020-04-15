.. _compound:

Compound Embedding
==================

A unique research contribution of TextWiser lies in its novel approach in creating embeddings from components, called the Compound Embedding. This method allows forming arbitrarily complex embeddings, thanks to a context-free grammar that defines a formal language for valid text featurization.

The compound embedding is instantiated using a schema which applies two main production rules:

* **Transform Operation:** This operator defines a list of operations. The first of these operations should be an ``Embedding`` while the rest should be ``Transformation(s)``. The idea is that the ``Embedding`` s have access to raw text and turn them into vectors, and therefore the following ``Transformation`` s need to operate on vectors. In PyTorch terms, this is equivalent to using ``nn.Sequential``.

* **Concatenation Operator:** This operator defines a concatenation of multiple embedding vectors. This can be done both at word and sentence level. In PyTorch terms, this is equivalent to using ``torch.cat``.

More formally, the compound schemas are defined by the following `context-free grammar <https://en.wikipedia.org/wiki/Context-free_grammar>`_:

.. _cfg:

A Context-Free Grammar of Embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    start → option | merge

    merge → "{" TRANSFORM ":" "[" start "," transform_list "]" "}"
          | "{" CONCAT ":" "[" concat_list "]" "}"

    transform_list → transform_start | transform_start "," transform_list

    concat_list → start | start "," concat_list

    transform_start → transform_option | "[" transform_option "," dict "]"

    option → embed_option | "[" embed_option "," dict "]"

    TRANSFORM → "transform"
    CONCAT → "concat"

    embed_option → BOW | DOC2VEC | TFIDF | USE | WORD | word_option

    BOW → "bow"
    DOC2VEC → "doc2vec"
    TFIDF → "tfidf"
    USE → "use"
    WORD → "word"

    word_option → FLAIR | CHAR | WORD2VEC | ELMO | BERT | GPT | GPT2 | TRANSFORMERXL | XLNET | XLM | ROBERTA

    FLAIR → "flair"
    CHAR → "char"
    WORD2VEC → "word2vec"
    ELMO → "elmo"
    BERT → "bert"
    GPT → "gpt"
    GPT2 → "gpt2"
    TRANSFORMERXL → "transformerXL"
    XLNET → "xlnet"
    XLM → "xlm"
    ROBERTA → "roberta"

    transform_option → LDA | NMF | POOL | SVD | UMAP

    LDA → "lda"
    NMF → "nmf"
    POOL → "pool"
    SVD → "svd"
    UMAP → "umap"

Example Compound Schema
^^^^^^^^^^^^^^^^^^^^^^^

Consider a compound embedding that achieves the following:
* creates a ``word2vec`` embedding which is then ``max`` pooled to document level
* creates a ``flair`` embedding  which is then ``mean`` pooled to document level
* creates a ``tfidf`` embedding reduced in dimensions using ``nmf``
* concatenates these three embeddings together
* decompose the concatenation via ``svd``

The :repo:`example schema </tree/master/notebooks/schema.json>` exactly captures this embedding:

.. code-block:: python

    example_schema = {
        "transform": [
            {
                "concat": [
                    {
                        "transform": [
                            ["word2vec", {"pretrained": "en"}],
                            "pool"
                        ]
                    },
                    {
                        "transform": [
                            ["flair", {"pretrained": "news-forward-fast"}],
                            ["pool", {"pool_option": "mean"}]
                        ]
                    },
                    {
                        "transform": [
                            "tfidf",
                            ["nmf", { "n_components": 30 }]
                        ]
                    }
                ]
            },
            "svd"
        ]
    }

    # Model: Compound
    emb = TextWiser(Embedding.Compound(schema=example_schema))

See the :repo:`usage example </tree/master/notebooks/basic_usage_example.ipynb>` for a runnable notebook with compound embedding.
