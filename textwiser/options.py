# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import copy
from enum import Enum
import os
from typing import Callable, Dict, List, NamedTuple, Optional, Union

from textwiser.utils import check_true, check_false, Constants, set_params


class WordOptions(Enum):
    """Supported Word embedding options.

    For more detail about the options, please consult the readme.
    """

    bytepair = "bytepair"
    char = "char"
    word2vec = "word2vec"
    flair = "flair"
    elmo = "elmo"
    bert = "bert"
    gpt = "gpt"
    gpt2 = "gpt2"
    transformerXL = "transformerXL"
    xlnet = "xlnet"
    xlm = "xlm"
    roberta = "roberta"
    distilbert = "distilbert"
    ctrl = "ctrl"
    albert = "albert"
    t5 = "t5"
    xlm_roberta = "xlm_roberta"
    bart = "bart"
    electra = "electra"
    dialo_gpt = "dialo_gpt"
    longformer = "longformer"

    def is_finetuneable(self):
        """Whether the Word embeddings are fine-tuneable"""
        return self not in (WordOptions.elmo,)

    def is_from_transformers(self):
        return self in (WordOptions.bert, WordOptions.gpt, WordOptions.gpt2, WordOptions.transformerXL,
                        WordOptions.xlnet, WordOptions.xlm, WordOptions.roberta, WordOptions.distilbert,
                        WordOptions.ctrl, WordOptions.albert, WordOptions.t5, WordOptions.xlm_roberta,
                        WordOptions.bart, WordOptions.electra, WordOptions.dialo_gpt, WordOptions.longformer)


class PoolOptions(Enum):
    """Supported pooling options.

    For more detail about the options, please consult the readme.
    """
    max = "max"
    min = "min"
    mean = "mean"
    first = "first"
    last = "last"


class _ArgBase:
    """
    Base class for specifying TextWiser arguments.

    Any implemented Embedding or Transformation model will have a corresponding container implementation that
    subclasses from _ArgBase. In order to implement a new model, you need to do the following:
    - Add a container to `options.py` under either `Embedding` or `Transformation`. If the model is finetuneable or if it
      can propogate gradients like `Transformation.Pool`, the container needs to implement the `_is_finetuneable` and
      `_can_backprop` functions because they will be false by default.
    - Add the container to either `Embedding_Type` or `Transformation_Type` in `options.py`.
    - Add the model implementation to either embeddings or transformations folders.
      - If there are any variables that you don't want to be saved alongside the model (such as temporary outputs),
        add their names to the `_ignored_vars` set in the implementation. See `_USEEmbeddings` as an example.
    - Import the implementation to `factory.py`, and add a mapping from the container class to the implementation class
      in the `factory` object.
    - If there's any more preprocessing needed (like how word options and pool options are handled), that needs to go
      either under the implementation itself, or under the `_get_and_init_doc_embeddings` function of `factory.py`.
      The latter is how word options and pool options are handled.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._frozen = True

    def _set(self, key, value):
        self.__setattr__(key, value, force=True)

    def __setattr__(self, key, value, force=False):
        if force or not hasattr(self, Constants.frozen) or not self._frozen:
            super().__setattr__(key, value)
        else:
            raise ValueError("Embeddings and Transformations are immutable!")

    def _get_attrs(self):
        return {k: v for k, v in self.__dict__.items() if k not in (Constants.kwargs, Constants.frozen)}

    def set_params(self, **kwargs):
        set_params(self, **kwargs)

    def _validate(self, finetune_enabled=False):
        pass

    def _is_finetuneable(self):
        return isinstance(self, (
            Embedding.Word,
            Transformation.SVD,
        ))

    def _can_backprop(self):
        return isinstance(self, (
            Transformation.Pool,
            Transformation.SVD,
        ))

    @staticmethod
    def from_string(model_name: str, params=None):
        classes = [c for c in _ArgBase.__subclasses__() if c != _EmbeddingBase] + [c for c in _EmbeddingBase.__subclasses__()]
        converter = {c.__name__.lower(): c for c in classes}
        if not params:
            params = dict()
        if model_name in converter:
            return converter.get(model_name)(**params)
        else:
            raise ValueError("The string embedding or transformation type %s is not supported." % model_name)


class _EmbeddingBase(_ArgBase):
    """
    Base class for specifying Embedding arguments.

    Attributes
    ----------
    pretrained: str
        The name or path of the pretrained model to use.
        If `None`, a new model will be trained from scratch when `fit` is called.
        Currently, TfIdf, BoW, Doc2Vec, and word2vec models are trainable;
        everything else will raise an error. Defaults to `default`, which signals
        the models to use the default pretrained model, or to train a new model if
        pretrained model is not available.
    """

    def __init__(self, pretrained=Constants.default_model, **kwargs):
        self.pretrained = pretrained
        super().__init__(**kwargs)


class Embedding(NamedTuple):
    class BOW(_EmbeddingBase):
        """Bag of words embeddings.

        Computes the frequency of words in each document and uses that as the representation.
        Supports any attributes that sklearn.feature_extraction.text.CountVectorizer may need.
        """
        def _validate(self, finetune_enabled=False):
            check_true(not self.pretrained or self.pretrained is Constants.default_model or
                       (isinstance(self.pretrained, str) and os.path.exists(self.pretrained)) or
                       hasattr(self.pretrained, 'read'),  # file-like
                       ValueError("The pretrained model should be a path to a pickle file or a file-like object."))

    class Compound(_EmbeddingBase):
        """Compound embeddings derived from a schema.

        Compound embeddings cannot be pretrained, you should instead save the whole TextWiser object.

        Attributes
        ----------
        schema: Union[Dict, list, tuple, str]
            The schema to load the models with.
            If it is a string and it is a valid path, it is assumed to be the path to a JSON file containing the schema.
            If it is a string and one of the supported embeddings or transformations, it will be used accordingly.
            If it is a list or a tuple, it is assumed to be one of the supported embeddings or transformations, and its parameters.
            If it is a dictionary object, it is assumed to be the JSON schema itself.
        pretrained: Union[NoneType, str]
            The pretrained model to use. Should be `None`, or be equal to `DEFAULT_MODEL`.
        """

        def __init__(self, schema: Union[Dict, list, tuple, str] = None, pretrained=Constants.default_model, **kwargs):
            self.schema = copy.deepcopy(schema)
            super().__init__(pretrained=pretrained, **kwargs)

        def _validate(self, finetune_enabled=False):
            check_true(isinstance(self.schema, (dict, str, tuple, list)),
                       TypeError("The schema should either be a dictionary, a valid embedding, an embedding-parameters tuple, or the path to a JSON file."))
            check_true(self.pretrained is None or self.pretrained is Constants.default_model,
                       ValueError("Compound embeddings cannot be pretrained. Save the TextWiser object instead."))

        def _is_finetuneable(self, schema=None):
            def can_backprop(transformation):
                if isinstance(transformation, (str, tuple, list)):
                    kwargs = dict()
                    if isinstance(transformation, (tuple, list)):
                        kwargs = transformation[1]
                        transformation = transformation[0]
                    return _ArgBase.from_string(transformation, kwargs)._can_backprop()

            schema = schema if schema else self.schema
            if isinstance(schema, (str, tuple, list)):
                kwargs = dict()
                if isinstance(schema, (tuple, list)):
                    kwargs = schema[1]
                    schema = schema[0]
                if schema in WordOptions.__members__:  # is a WordOption
                    return Embedding.Word(word_option=WordOptions[schema], **kwargs)._is_finetuneable()
                return _ArgBase.from_string(schema, kwargs)._is_finetuneable()

            if 'transform' in schema:
                transforms = schema['transform']
                for transformation in transforms[:0:-1]:
                    if self._is_finetuneable(schema=transformation):
                        return True
                    elif not can_backprop(transformation):
                        return False
                return self._is_finetuneable(schema=transforms[0])

            if 'concat' in schema:
                concats = schema['concat']
                return any([self._is_finetuneable(concat) for concat in concats])

    class Doc2Vec(_EmbeddingBase):
        """Doc2Vec embeddings.

        These embeddings extend the word embeddings to documents using a similar (CBOW) approach.
        Supports any attributes that gensim.models.doc2vec.Doc2Vec may need.

        Attributes
        ----------
        deterministic: bool
            Flag for making the inference procedure deterministic.
            Due to sampling used in the inference process, multiple calls with the same input
            results in different vectors. Setting this flag makes the inference deterministic,
            at the cost of speed. Defaults to false.
        tokenizer: Callable
            A tokenizer function that takes in a document string and returns
            a list of tokens. Defaults to whitespace splitting.
        pretrained: str
            The name or path of the pretrained model to use.
            If `None`, a new model will be trained from scratch when `fit` is called.
            Currently, TfIdf, BoW, Doc2Vec, and word2vec models are trainable;
            everything else will raise an error. Defaults to `default`, which signals
            the models to use the default pretrained model, or to train a new model if
            pretrained model is not available.
        """

        def __init__(self, deterministic: bool = False, tokenizer: Optional[Callable[[str], List[str]]] = None,
                     pretrained=Constants.default_model, **kwargs):
            self.deterministic = deterministic
            self.tokenizer = tokenizer
            super().__init__(pretrained=pretrained, **kwargs)

        def _validate(self, finetune_enabled=False):
            check_true(isinstance(self.deterministic, bool), TypeError("The deterministic parameter should be a boolean."))
            if self.tokenizer:
                doc = "string"
                res = self.tokenizer(doc)
                check_true(isinstance(res, list), TypeError("The tokenizer should return a list of tokens."))
                check_true(isinstance(res[0], str), TypeError("The tokens should be of string type."))
            check_true(not self.pretrained or self.pretrained is Constants.default_model or
                       (isinstance(self.pretrained, str) and os.path.exists(self.pretrained)) or
                       hasattr(self.pretrained, 'read'),  # file-like
                       ValueError("The pretrained model should be a path to a pickle file or a file-like object."))

    class Random(_EmbeddingBase):
        """Randomized configuration of embeddings.

        Creates a random schema, including the possibility of having transformations and concatenations
        (and even recursive ones). The only restriction is that word vectors need to be pooled immediately within a transformation,
        whereas they can theoretically be concatenated first and pooled later.

        The schemas generated are biased towards sensible values, so as not to create huge models and not waste time
        with irrelevant configurations. However it is still possible to get large models and even models that do not work,
        such as a word pooling (which has negative values) followed by NMF (which expects nonnegative values).
        """
        def _validate(self, finetune_enabled=False):
            check_true(self.pretrained is None or self.pretrained is Constants.default_model,
                       ValueError("Random embeddings cannot be pretrained. Save the TextWiser object instead."))

    class TfIdf(_EmbeddingBase):
        """TfIdf embeddings.

        Computes the frequency of words in each document, balances them with the inverse document frequency,
        and uses that as the representation. Supports any attributes that
        sklearn.feature_extraction.text.TfidfVectorizer may need.
        """
        def _validate(self, finetune_enabled=False):
            check_true(not self.pretrained or self.pretrained is Constants.default_model or
                       (isinstance(self.pretrained, str) and os.path.exists(self.pretrained)) or
                       hasattr(self.pretrained, 'read'),  # file-like
                       ValueError("The pretrained model should be a path to a pickle file or a file-like object."))

    class USE(_EmbeddingBase):
        """Universal Sentence Encoder embeddings.

        Uses Google's Universal Sentence Encoder model, which is in Tensorflow. As such,
        Tensorflow is a required dependency for using this model. Since the model is pre-built,
        there are no parameters that can be passed in.
        """
        def _validate(self, finetune_enabled=False):
            import tensorflow  # this will fail if tensorflow is not available
            import tensorflow_hub  # this will fail if tensorflow_hub is not available
            check_true(self.pretrained, ValueError("USE needs to be pretrained."))

    class Word(_EmbeddingBase):
        """Word embeddings.

        These include non-contextual Word2Vec models, such as the original skip-gram model,
        and the GloVe embeddings, and also contextual embeddings, such as BERT and ELMo.
        The list of all supported embeddings are in WordOptions, which includes
        WordOptions.word2vec to support all models available in the Flair framework.

        Attributes
        ----------
        word_option: WordOptions
            The word vectorization model to use.
            Defaults to WordOptions.word2vec.
        pretrained: str
            The name or path of the pretrained model to use.
            If ``None``, a new model will be trained from scratch when `fit` is called.
            Currently, only a word2vec model can be trained, everything else will raise an error.
            Defaults to ``default``, which uses the Flair defaults for each model.
        sparse: bool
            Whether the embeddings layer should be implemented sparsely.
            Only used with word2vec embeddings. Sparse embeddings are useful for backpropagation,
            where only a fraction of the embeddings need to be updated. Specifying it results in a
            significant speedup in the backward call. However, keep in mind that only a limited number
            of optimizers support sparse gradients: currently it’s optim.SGD (CUDA and CPU),
            optim.SparseAdam (CUDA and CPU) and optim.Adagrad (CPU).
        tokenizer: Callable
            A tokenizer function that takes in a document string and returns
            a list of tokens. Defaults to whitespace splitting. Only used with word2vec embedings.
        layers: Union[int, List[int]]
            The hidden layers to use as the word representation in transformers models.
            Defaults to ``-1``.
        inline_pool_option: Optional[PoolOptions]
            The pooling to be done right after the word embedding for each document (``inline``). This is the same as
            having a word embedding immediately followed by a pool transformation.

            Specifying a pool transformation here can be significantly less memory intensive if you're embedding a large
            dataset at once. If specified, the return value will either be a numpy array or a torch tensor (depending on
            specified dtype) instead of a list of arrays or tensors. Specifying both ``inline_pool_option`` and a
            following ``Transformation.Pool`` for a single Word embedding will result in an error.

            Defaults to ``None``, which means no pooling will be done.
        """

        def __init__(self, word_option: WordOptions = WordOptions.word2vec, pretrained: str = Constants.default_model,
                     sparse: bool = False, tokenizer: Optional[Callable[[str], List[str]]] = None,
                     layers: Union[int, List[int]] = -1, inline_pool_option: Optional[PoolOptions] = None, **kwargs):
            self.word_option = word_option
            self.sparse = sparse
            self.tokenizer = tokenizer
            self.layers = layers
            self.inline_pool_option = inline_pool_option
            super().__init__(pretrained=pretrained, **kwargs)

        def _validate(self, finetune_enabled=False):
            check_true(isinstance(self.word_option, WordOptions),
                       ValueError("The embedding must be one of the supported word embeddings."))
            check_true(self.pretrained or self.word_option is WordOptions.word2vec,
                       ValueError("Only word2vec embeddings can be trained from scratch."))
            check_true(not finetune_enabled or self._is_finetuneable(),
                       ValueError("The weights can only be fine-tuned if they are not ELMo embeddings."))
            check_false(not finetune_enabled and self.word_option == WordOptions.char,
                        ValueError("Character embeddings are only available if the model is fine-tuneable."))
            check_true(not self.sparse or self.word_option == WordOptions.word2vec,
                       ValueError("Sparse embeddings only supported with word2vec embeddings"))
            check_true(isinstance(self.layers, int) or all([isinstance(l, int) for l in self.layers]),
                       ValueError("Layers can only be an integer or a list of integers"))
            check_true(not self.inline_pool_option or isinstance(self.inline_pool_option, PoolOptions),
                       ValueError("Inline pooling should either be None or a pool option."))

            if self.word_option == WordOptions.elmo:
                import tensorflow  # this will fail if tensorflow is not available
                import tensorflow_hub  # this will fail if tensorflow_hub is not available
                check_true(self.pretrained, ValueError("ELMo needs to be pretrained."))

            if self.tokenizer:
                check_true(self.word_option == WordOptions.word2vec,
                           ValueError("The tokenizer can only be used if word2vec embeddings are used."))
                doc = "string"
                res = self.tokenizer(doc)
                check_true(isinstance(res, list), TypeError("The tokenizer should return a list of tokens."))
                check_true(isinstance(res[0], str), TypeError("The tokens should be of string type."))

        def _is_finetuneable(self):
            return self.word_option.is_finetuneable()


class Transformation(NamedTuple):
    class LDA(_ArgBase):
        """Latent Dirichlet Allocation Transformation.

        Finds topics with a Dirichlet prior and converts vectors into topic distributions.
        Supports any attributes that sklearn.decomposition.LatentDirichletAllocation may need.

        Attributes
        ----------
        n_components: int
            The number of topics to use.
        """

        def __init__(self, n_components: int = 10, **kwargs):
            self.n_components = n_components
            super().__init__(**kwargs)

        def _validate(self, finetune_enabled=False):
            check_true(isinstance(self.n_components, int), TypeError("The number of components must be an integer."))
            check_true(self.n_components >= 2, ValueError("The number of components must be at least two."))

    class NMF(_ArgBase):
        """NMF Transformation.

        Decomposes one nonnegative matrix H into nonnegative matrices U and V such that
        H = UV^T. Supports any attributes that sklearn.decomposition.NMF may need.

        Attributes
        ----------
        n_components: int
            The number of factors to find.
        """

        def __init__(self, n_components: int = 10, **kwargs):
            self.n_components = n_components
            super().__init__(**kwargs)

        def _validate(self, finetune_enabled=False):
            check_true(isinstance(self.n_components, int), TypeError("The number of components must be an integer."))
            check_true(self.n_components >= 2, ValueError("The number of components must be at least two."))

    class UMAP(_ArgBase):
        """Uniform Manifold Approximation & Projection Transformation

        Tries to model the manifold with a fuzzy topological structure.
        Supports any attributes that umap.UMAP may need.

        Attributes
        ----------
        n_components: int
            The number of topics to use.
        random_state: Optional[int]
            The random number seed. Use ``None`` to get non-reproducible results, which might be useful for
            testing purposes.
        deterministic_init: bool
            Whether to initialize UMAP deterministically across operating systems. If ``False``, training a UMAP
            model in MacOS will lead to a different set of embeddings than in Ubuntu. If ``True``, the generated
            embeddings will be the same. Requires ``random_state`` to be set to a value if ``True``.
        """

        def __init__(self, n_components: int = 10, random_state: Optional[int] = Constants.default_seed,
                     deterministic_init: bool = True, **kwargs):
            self.n_components = n_components
            self.random_state = random_state
            self.deterministic_init = deterministic_init
            super().__init__(**kwargs)

        def _validate(self, finetune_enabled=False):
            import umap  # this will fail if umap is not available
            check_true(isinstance(self.n_components, int), TypeError("The number of components must be an integer."))
            check_true(self.n_components >= 2, ValueError("The number of components must be at least two."))
            check_true(isinstance(self.n_components, int), TypeError("The number of components must be an integer."))
            check_true(self.random_state is None or isinstance(self.random_state, int),
                       TypeError("The random seed must be an integer or ``None``."))
            check_true(isinstance(self.deterministic_init, bool),
                       TypeError("The deterministic init parameter must be a boolean."))
            check_true(not self.deterministic_init or self.random_state is not None,
                       ValueError("If deterministic init parameter is ``True``, the random state must also be "
                                  "specified."))
            check_true(not self.deterministic_init or 'init' not in self.kwargs,
                       ValueError("If deterministic init parameter is ``True``, you cannot pass an ``init``"
                                  "parameter to UMAP"))

    class Pool(_ArgBase):
        """Pool Transformation.

        Pools a list of word embeddings into a single embedding. Should not be specified if ``inline_pool_option``
        is specified for the corresponding word embedding.

        Attributes
        ----------
        pool_option: PoolOptions
            The type of pooling operation. Most common is max, which takes
            the maximum values for all features across the words.
        """

        def __init__(self, pool_option: PoolOptions = PoolOptions.max, **kwargs):
            self.pool_option = pool_option
            super().__init__(**kwargs)

        def _validate(self, finetune_enabled=False):
            check_true(isinstance(self.pool_option, PoolOptions),
                       TypeError("The pool type must be models.options.PoolOptions"))

    class SVD(_ArgBase):
        """SVD Transformation.

        Decomposes a matrix M into matrices U, S, and V such that M=USV^T.
        S becomes the matrix of singular values, while U and V matrices are
        the instance and feature representations using singular values as features.
        Supports any attributes that sklearn.decomposition.SVD may need.

        Attributes
        ----------
        n_components: int
            The number of factors to find.
        """

        def __init__(self, n_components: int = 10, **kwargs):
            self.n_components = n_components
            super().__init__(**kwargs)

        def _validate(self, finetune_enabled=False):
            check_true(isinstance(self.n_components, int), TypeError("The number of components must be an integer."))
            check_true(self.n_components >= 2, ValueError("The number of components must be at least two."))


Embedding_Type = Union[Embedding.BOW,
                       Embedding.Compound,
                       Embedding.Doc2Vec,
                       Embedding.Random,
                       Embedding.TfIdf,
                       Embedding.USE,
                       Embedding.Word]


Transformation_Type = Union[Transformation.LDA,
                            Transformation.NMF,
                            Transformation.Pool,
                            Transformation.SVD,
                            Transformation.UMAP]
