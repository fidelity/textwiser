.. _transformations:

Transformations
===============

.. csv-table::
    :header: "Transformations", "Notes"

    "`Singular Value Decomposition (SVD) <https://pytorch.org/docs/stable/torch.html#torch.svd>`_", "| Differentiable"
    "`Latent Dirichlet Allocation (LDA) <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation>`_", "| Not differentiable"
    "`Non-negative Matrix Factorization (NMF) <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF>`_",  "| Not differentiable"
    "`Uniform Manifold Approximation and Projection (UMAP) <https://umap-learn.readthedocs.io/en/latest/parameters.html>`_", "| Not differentiable"
    "Pooling Word Vectors", "| Applies to word embeddings only
    | Reduces word-level vectors to document-level
    | Pool options include ``max``, ``min``, ``mean``, ``first``, and ``last``
    | Defaults to ``max``"
