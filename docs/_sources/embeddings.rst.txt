.. _embeddings:

Embeddings
============

.. csv-table::
    :header: "Embeddings", "Notes"

    "`Bag of Words (BoW) <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer>`_", "| Supported by ``scikit-learn``
    | Defaults to training from scratch"
    "`Term Frequency Inverse Document Frequency (TfIdf) <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_", "| Supported by ``scikit-learn``
    | Defaults to training from scratch"
    "`Document Embeddings (Doc2Vec) <https://radimrehurek.com/gensim/models/doc2vec.html>`_", "| Supported by ``gensim``
    | Defaults to training from scratch"
    "`Universal Sentence Encoder (USE) <https://tfhub.dev/google/universal-sentence-encoder-large/5>`_", "| Supported by ``tensorflow``, see :ref:`requirements<requirements>`
    | Defaults to `large v5 <https://tfhub.dev/google/universal-sentence-encoder-large/5>`_"
    ":ref:`compound`", "| Supported by a :ref:`context-free grammar<cfg>`"
    "Word Embedding: `Word2Vec <https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md>`_", "| Supported by these `pretrained embeddings <https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md>`_
    | Common pretrained options include ``crawl``, ``glove``, ``extvec``, ``twitter``, and ``en-news``
    | When the pretrained option is ``None``, trains a new model from the given data
    | Defaults to ``en``, FastText embeddings trained on news"
    "Word Embedding: `Character <https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_3_WORD_EMBEDDING.md#character-embeddings>`_", "| Initialized randomly and not pretrained
    | Useful when trained for a downstream task
    | Enable :ref:`fine-tuning<fine_tuning>` to get good embeddings"
    "Word Embedding: `BytePair <https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md>`_ ", "| Supported by these `pretrained embeddings <https://nlp.h-its.org/bpemb/#download>>`_
    | Pretrained options can be specified with the string ``<lang>_<dim>_<vocab_size>``
    | Default options can be omitted like ``en``, ``en_100``, or ``en__10000``
    | Defaults to ``en``, which is equal to ``en_100_10000``"
    "Word Embedding: `ELMo <https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/ELMO_EMBEDDINGS.md>`_", "| Supported by these `pretrained embeddings <https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/ELMO_EMBEDDINGS.md>`_ from `AllenNLP <https://allennlp.org>`_
    | Defaults to ``original``"
    "Word Embedding: `Flair <https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md>`_", "| Supported by these `pretrained embeddings <https://github.com/zalandoresearch/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md>`_
    | Defaults to ``news-forward-fast``"
    "Word Embedding: `BERT <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``bert-base-uncased``"
    "Word Embedding: `OpenAI GPT <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``openai-gpt``"
    "Word Embedding: `OpenAI GPT2 <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``gpt2-medium``"
    "Word Embedding: `TransformerXL <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``transfo-xl-wt103``"
    "Word Embedding: `XLNet <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``xlnet-large-cased``"
    "Word Embedding: `XLM <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``xlm-mlm-en-2048``"
    "Word Embedding: `RoBERTa <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``roberta-base``"
    "Word Embedding: `DistilBERT <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``distilbert-base-uncased``"
    "Word Embedding: `CTRL <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``ctrl``"
    "Word Embedding: `ALBERT <https://github.com/huggingface/transformers#model-architectures>`_", "| Supported by these `pretrained embeddings <https://huggingface.co/transformers/pretrained_models.html>`_
    | Defaults to ``albert-base-v2``"

Tokenization
^^^^^^^^^^^^

In general, text data should be **whitespace-tokenized** before being fed into TextWiser.

* The ``BOW``, ``Doc2Vec``, ``TfIdf`` and ``Word`` embeddings also accept an optional ``tokenizer`` parameter.
* The ``BOW`` and ``TfIdf`` embeddings expose all the functionality of the underlying scikit-learn models, so it is also possible to specify other text preprocessing options such as ``stop_words``.
* Tokenization for ``Doc2Vec`` and ``Word`` splits using whitespace. The latter model only uses the ``tokenizer`` parameter if the ``word_options`` parameter is set to ``WordOptions.word2vec``, and will raise an error otherwise.
