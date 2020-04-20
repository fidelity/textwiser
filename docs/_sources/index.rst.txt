TextWiser: Text Featurization Library
=====================================

TextWiser is a research library that provides a unified framework for text featurization based on a rich set of methods
while taking advantage of pretrained models provided by the state-of-the-art `Flair <https://github.com/zalandoresearch/flair>`_ library.

The main contributions include:

* **Rich Set of Embeddings:** A wide range of available :ref:`embeddings` and :ref:`transformations` to choose from.

* **Fine-Tuning:** Designed to support a ``PyTorch`` backend, and hence, retains the ability to :ref:`fine-tune<fine_tuning>` for downstream tasks. That means, if you pass the resulting fine-tunable embeddings to a training method, the features will be optimized automatically for your application.

* **Parameter Optimization:** Interoperable with the standard ``scikit-learn`` pipeline for hyper-parameter tuning and rapid experimentation. All underlying parameters are exposed to the user.

* **Grammar of Embeddings:** Introduces a novel approach to design embeddings from components.  The :ref:`compound embedding<compound>` allows forming arbitrarily complex embeddings in accordance with a :ref:`context-free grammar<cfg>` that defines a formal language for valid text featurization.

* **GPU Native:** Built with GPUs in mind. If it detects available hardware, the relevant models are automatically placed on the GPU.

TextWiser is developed by the Artificial Intelligence Center of Excellence at Fidelity Investments.

.. include:: quick.rst

Source Code
===========
The source code is hosted on :repo:`GitHub <>`.

.. sidebar:: Contents

   .. toctree::
    :maxdepth: 2

    quick
    installation
    examples
    embeddings
    transformations
    compound
    fine_tuning
    contributing
    api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
