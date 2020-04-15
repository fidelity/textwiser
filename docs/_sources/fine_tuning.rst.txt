.. _fine_tuning:

Fine-Tuning for Downstream Tasks
================================

All Word2Vec and transformer-based embeddings and any embedding followed with an ``svd`` transformation are fine-tunable for downstream tasks. In other words, if you pass the resulting fine-tunable embedding to a PyTorch training method, the features will automatically be trained for your application.

The fine-tuning is disabled by default. To activate it, turn the ``is_finetuneable`` parameter on and specify a torch ``dtype`` as the type of the output.

.. code-block:: python

    emb = TextWiser(Embedding.Word(word_option=WordOptions.word2vec),
                    Transformation.Pool(pool_option=PoolOptions.max),
                    is_finetuneable=True, dtype=torch.float32)


Notice also that setting the ``sparse`` parameter of a ``WordOptions.word2vec`` model to ``True`` can yield significant speedup during training. Currently, ``optim.SGD`` (CUDA and CPU), ``optim.SparseAdam`` (CUDA and CPU), and ``optim.Adagrad`` (CPU) support sparse embeddings.

:repo:`Fine Tuning Example </tree/master/notebooks/finetune_example.ipynb>` shows the ability to fine-tune embeddings, and how they can improve the prediction performance.
