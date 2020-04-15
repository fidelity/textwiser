# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

import abc

import torch.nn as nn

from textwiser.utils import set_params


class BaseFeaturizer(abc.ABC, nn.Module):
    @abc.abstractmethod
    def fit(self, x, y=None):
        """Trains the model with the given input data.

        Parameters
        ----------
        x: Union[str, List[str]]
            The training documents to fit the model on.
        y: Optional
            An optional target argument for possibly doing supervised training.
            There are currently no supervised models in TextWiser, but the option
            to do so is here for the future. This also allows TextWiser to fit
            into the sklearn pipelines.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, x):
        """Transforms the data into embeddings.

        Parameters
        ----------
        x: Union[str, List[str]]
            The documents to transform into embeddings.
        """
        raise NotImplementedError()

    def transform(self, x):
        """Transforms the data into embeddings.

        Subclasses don't need to implement this method. Instead, they should implement
        `forward`, which implicitly gets called when using `transform`, and integrates
        fully with PyTorch.

        Parameters
        ----------
        x: Union[str, List[str]]
            The documents to transform into embeddings.
        """
        return self(x)

    def fit_transform(self, x, y=None):
        """Trains the model with the given input data, and transforms the data into embeddings.

        This method is equivalent to calling `fit` first and `transform` later.

        Parameters
        ----------
        x: Union[str, List[str]]
            The training documents to fit the model on, and to transform into embeddings.
        y: Optional
            An optional target argument for possibly doing supervised training.
            There are currently no supervised models in TextWiser, but the option
            to do so is here for the future. This also allows TextWiser to fit
            into the sklearn pipelines.
        """
        self.fit(x, y)
        return self.transform(x)

    def set_params(self, **kwargs):
        """Convenience method for allowing Scikit-learn style setting of parameters.

        This is especially useful when using the randomized or grid search model selection
        classes of Scikit-learn. See the `pipeline_example.ipynb` notebook for an example.
        """
        set_params(self, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """PyTorch state dictionary enhanced with the ability to keep any attribute.

        By default, any variable created inside an algorithm implementation (derived from the BaseFeaturizer) will
        be automatically pickled and put into the state dictionary. This is in contrast to the PyTorch behaviour, where
        only nn.Modules or nn.Parameters are placed into the state dictionary. Since not all of our models are
        PyTorch models, this implementation decreases the work that needs to be done. This makes it easy to interoperate
        with `load_state_dict`.

        If there are variables inside the objects that shouldn't be saved, their names can be placed in the
        `_ignored_args` parameter. See the `_USEEmbeddings` class as example.
        """
        sd = super(BaseFeaturizer, self).state_dict(destination, prefix, keep_vars)
        ignored_args = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks',
                        '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_modules',
                        'training'}
        try:
            ignored_args |= self._ignored_args
        except AttributeError:  # There is no `_ignored_args`
            pass
        saved_args = [k for k in self.__dict__.keys() if k not in ignored_args]
        for varname in saved_args:
            sd[prefix + varname] = getattr(self, varname)
        sd[prefix + '_saved_args'] = saved_args
        return sd

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Overridden PyTorch state dict loading for any attribute."""
        saved_args = prefix + '_saved_args'
        if saved_args in state_dict:
            for varname in state_dict[saved_args]:
                setattr(self, varname, state_dict[prefix + varname])
                state_dict.pop(prefix + varname)
            state_dict.pop(saved_args)
        super(BaseFeaturizer, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
