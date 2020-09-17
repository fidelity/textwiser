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

    start → embed_like | merge

    embed_like → embed_option | "[" embed_option "," dict "]"

    embed_option → BOW | DOC2VEC | TFIDF | USE

    merge → "{" TRANSFORM ":" "[" start "," transform_list "]" "}"
          | "{" TRANSFORM ":" "[" word_like "," pool_transform_list "]" "}"
          | "{" CONCAT ":" "[" concat_list "]" "}"

    transform_list → transform_like | transform_like "," transform_list

    transform_like → transform_option | "[" transform_option "," dict "]"

    transform_option → LDA | NMF | SVD | UMAP

    word_like → WORD
              | "[" WORD "," dict "]"
              | word_option
              | "[" word_option "," dict "]"

    word_option → FLAIR | CHAR | WORD2VEC | ELMO | BERT | GPT | GPT2 | TRANSFORMERXL | XLNET | XLM | ROBERTA | DISTILBERT | CTRL | ALBERT | T5 | XLM_ROBERTA | BART | ELECTRA | DIALO_GPT | LONGFORMER

    pool_transform_list → pool_like
                        | pool_like "," transform_list
                        | transform_list "," pool_like
                        | transform_list "," pool_like "," transform_list

    pool_like → POOL | "[" POOL "," dict "]"

    concat_list → start | start "," concat_list

    TRANSFORM → "transform"
    CONCAT → "concat"

    BOW → "bow"
    DOC2VEC → "doc2vec"
    TFIDF → "tfidf"
    USE → "use"
    WORD → "word"

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
    DISTILBERT → "distilbert"
    CTRL → "ctrl"
    ALBERT → "albert"
    T5 → "t5"
    XLM_ROBERTA → "xlm_roberta"
    BART → "bart"
    ELECTRA → "electra"
    DIALO_GPT → "dialo_gpt"
    LONGFORMER → "longformer"

    LDA → "lda"
    NMF → "nmf"
    POOL → "pool"
    SVD → "svd"
    UMAP → "umap"

This grammar captures the universe of valid configurations for embeddings that can be specified in TextWiser.
Note that the ``dict`` non-terminal denotes a valid JSON dictionary, but is left outside this definition for the sake of brevity.
A sample implementation of ``dict`` can be found `here <https://github.com/lark-parser/lark/blob/master/docs/json_tutorial.md>`_.

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
