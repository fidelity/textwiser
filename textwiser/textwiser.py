# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

from typing import List, Union
import warnings

import numpy as np
import torch

from textwiser.base import BaseFeaturizer
from textwiser.factory import get_standalone_document_embeddings, _Sequential
from textwiser.options import _ArgBase, Embedding, Embedding_Type, Transformation, Transformation_Type
from textwiser.utils import check_true, check_false, convert, device, OutputType


class TextWiser(BaseFeaturizer):
    """**TextWiser: Text Featurization Library**

    TextWiser is a library that provides a unified framework for text featurization based on a rich set of methods.

    Attributes
    ----------
    _imp: _Sequential
        The model pipeline with one embedding and zero or more transformations.
        The text featurization starts with an embedding, and is sequentially followed
        by the transformations. Note that the embedding can also be a Compound embedding,
        which may be a combination of multiple embeddings.
    dtype: Union[np.generic, torch.dtype]
        The requested type of the output.
        This is useful when working with a downstream task that only uses NumPy,
        and allows TextWiser to fit into the scikit-learn Pipeline. Defaults to a pytorch tensor.
    """

    def __init__(self,
                 embedding: Embedding_Type,
                 transformations: Union[None,
                                        Transformation_Type,
                                        List[Transformation_Type]] = None,
                 is_finetuneable: bool = False,
                 dtype: Union[np.generic, torch.dtype] = np.float32,
                 lazy_load: bool = False):
        """Initializes with the given arguments.

        Parameters
        ----------
        embedding: Embedding_Type
            The type of document embedding to use.
            These types are defined under options.Embedding
            At least 1 is necessary to create a valid featurization.
        transformations: Union[None, Transformation_Type, List[Transformation_Type]]
            The transformations to be applied on top of the embedding.
            There can be 0, 1, or many transformations following a single embedding.
            Defaults to no transformations.
        is_finetuneable: bool
            Whether the model should be fine-tuneable. If set, the model weights will be
            updated when optimized with PyTorch in a downstream task. If the input corpus is small,
            there is a chance that this will cause the model the overfit. In that case, it is best to keep
            the model weights frozen. Defaults to False.
        dtype: Union[np.generic, torch.dtype]
            The requested type of the output.
            Since TextWiser fits into both Scikit-learn and PyTorch use-cases, it can output both numpy arrays
            and torch tensors. Note that dtype has to be a torch tensor for the model to be fine-tuneable.
            Defaults to a numpy array.
        lazy_load: bool
            Whether to initialize the model parameters at usage time.
            If set, the model parameters such as word embeddings will only be loaded
            into the memory when `fit` or `forward` is called. If not, they will be
            loaded when __init__ is called.
        """
        super(TextWiser, self).__init__()

        if transformations:
            if isinstance(transformations, _ArgBase):
                transformations = [transformations]

        # Validate arguments
        TextWiser._validate_init_args(embedding, transformations, is_finetuneable, dtype)

        # Save arguments
        self.embedding = embedding
        self.transformations = transformations
        self.is_finetuneable = is_finetuneable
        self.dtype = dtype
        self.is_fitted = False

        # Create the model
        self._imp = None
        if not lazy_load:
            self._init_all_models()

    def _init_all_models(self):
        models = [self._init_model(self.embedding)]
        if self.transformations:
            models.extend([self._init_model(transformation) for transformation in self.transformations])
        self._imp = _Sequential(*models)
        self.to(device)

    def _init_model(self, model: Union[Embedding_Type, Transformation_Type]):
        return get_standalone_document_embeddings(model)

    def fit(self, x=None, y=None):
        """Trains the model with the given input data.

        Parameters
        ----------
        x: Union[None, str, List[str]]
            The training documents to fit the model on.
        y: Optional
            An optional target argument for possibly doing supervised training.
            There are currently no supervised models in TextWiser, but the option
            to do so is here for the future. This also allows TextWiser to fit
            into the sklearn pipelines.
        """
        self._validate_fit_args(x, y)
        if not self._imp:  # Lazy loaded
            self._init_all_models()
        self.train()
        self._imp.fit(self._standardize_input(x), y)
        self.is_fitted = True
        return self

    def _standardize_input(self, x):
        if isinstance(x, str):
            return [x]
        elif x is None:
            return []
        return x

    def _convert_to_dtype(self, x, detach=False):
        typ = OutputType.tensor if isinstance(self.dtype, torch.dtype) else OutputType.array
        return convert(x, typ, self.dtype, detach=detach)

    def forward(self, x):
        """Transforms the data into embeddings.

        If `is_finetuneable` is False, the model is turned into the eval mode
        and gradients will not be propagated. If True, no assumptions about the mode
        or the gradients will be made.

        Parameters
        ----------
        x: Union[str, List[str]]
            The documents to transform into embeddings.
        """
        if self.is_fitted:
            x = self._standardize_input(x)
            if self.is_finetuneable:
                return self._forward(x)
            else:
                self.eval()
                with torch.no_grad():
                    return self._forward(x, detach=True)
        else:
            raise NotImplementedError("You must call the `fit` method before featurizing the text.")

    def _forward(self, x, detach=False):
        return self._convert_to_dtype(self._imp(x), detach=detach)

    def fit_transform(self, x, y=None):
        self._validate_fit_args(x, y)
        if not self._imp:  # Lazy loaded
            self._init_all_models()
        if self.is_finetuneable:
            self.train()
            x = self._convert_to_dtype(self._imp.fit_transform(self._standardize_input(x), y))
        else:
            self.eval()
            with torch.no_grad():
                x = self._convert_to_dtype(self._imp.fit_transform(self._standardize_input(x), y), detach=True)
        self.is_fitted = True
        return x

    @staticmethod
    def _check_finetuneable(embedding, transformations):
        if transformations:
            for transformation in transformations[::-1]:
                if transformation._is_finetuneable():
                    return True
                elif not transformation._can_backprop():
                    return False
        return embedding._is_finetuneable()

    @staticmethod
    def _validate_init_args(embedding, transformations, is_finetuneable, dtype):
        """
        Validates arguments for the constructor.
        """

        # Embedding
        embedding._validate(finetune_enabled=is_finetuneable)

        # Transformation
        if transformations:
            [transformation._validate() for transformation in transformations]

        # words should be pooled
        if isinstance(embedding, Embedding.Word) and embedding.inline_pool_option is None and (
                not transformations or
                not any([isinstance(transformation, Transformation.Pool) for transformation in transformations])):
            warnings.warn("Word embeddings are specified but no pool options are specified. Are you sure you don't want to pool them?", RuntimeWarning)

        # words shouldn't be double-pooled
        check_false(isinstance(embedding, Embedding.Word) and embedding.inline_pool_option is not None and transformations
                    and any([isinstance(transformation, Transformation.Pool) for transformation in transformations]),
                    ValueError("You cannot specify both `inline_pool_option` and `Pool` transformation for the same"
                               " embedding at the same time. Please pick one!"))

        # dtype
        check_true(isinstance(dtype, torch.dtype) or issubclass(dtype, np.generic),
                   TypeError("The dtype must be either a numpy or torch type."))
        check_true(not is_finetuneable or isinstance(dtype, torch.dtype),
                   TypeError("The dtype must be torch for model to be fine-tuneable."))
        check_true(not is_finetuneable or TextWiser._check_finetuneable(embedding, transformations),
                   ValueError("Model must have fine-tuneable weights if `is_finetuneable` is specified."))

    @staticmethod
    def _validate_fit_args(x, y):
        pass
