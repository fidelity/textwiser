.. _installation:

Installation
============

.. admonition:: Installation Options

    There are two alternatives to install the library:

    1. Install from PyPI using the prebuilt wheel package (``pip install textwiser``)
    2. Build from the source code

.. _requirements:

Requirements
------------

The library requires Python **3.8+**. The ``requirements.txt`` lists the necessary packages.
It is **strongly recommended** to install PyTorch following the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_ before installing TextWiser.
Otherwise, you might get stuck with a CPU-only version.
Once PyTorch is installed, you can install the requirements using ``pip install -r requirements.txt``.

The library is based on PyTorch but it also relies on:

* Scikit-learn for NMF, LDA, TfIdf, and BoW
* Flair for word vectors
* Transformers for contextual word vectors
* Spacy and it's ``en`` model are optional imports for OpenAI GPT; the model can be installed using ``python -m spacy download en``
* Tensorflow is an optional import for Universal Sentence Encoder and ELMo. If you want to use USE or ELMo, make sure you satisfy ``tensorflow>=2.0.0`` and ``tensorflow-hub>=0.7.0``.
* UMAP is an optional import for UMAP transformation. If you want to use UMAP, make sure you satisfy ``umap-learn>=0.5.1``

PyPI
----

TextWiser can be installed using ``pip install textwiser``, which will download the latest wheel from
`PyPI <http://pypi.org/project/textwiser/>`_. This will also install all required dependencies.

Alternatively, you can use ``pip install textwiser[full]`` to install TextWiser with all the optional dependencies.

Source Code
-----------

Alternatively, you can build a wheel package on your platform from scratch using the source code:

.. code-block:: bash

    git clone https://github.com/fidelity/textwiser
    cd textwiser
    pip install setuptools wheel # if wheel is not installed
    python setup.py bdist_wheel
    pip install dist/textwiser-X.X.X-py3-none-any.whl

.. important:: Don't forget to replace ``X.X.X`` with the current version number.

Test Your Setup
---------------
To confirm that installing the package was successful, run the first example in the :ref:`Quick Start<quick>`. To confirm that the whole installation was successful, run the tests and all should pass. When running the tests, it will download a 50MB pretrained model. Note that the ``PYTHONHASHSEED=0`` variable is necessary to ensure Doc2Vec training is reproducible - you do not need this if reproducibility is not important, or if you're not using Doc2Vec.

.. code-block:: bash

    PYTHONHASHSEED=0 python -m unittest discover -v tests

You can also set the ``TEST_WORD_EMBEDDINGS`` environmental variable to comma-separated word embeddings (ex: ``bert,flair``) to test them, or to ``all`` to test all possible word embeddings. Note that this will download all word embeddings, which is very time-consuming, and it assumes all optional requirements are satisfied.

.. code-block:: bash

    TEST_WORD_EMBEDDINGS=all python -m unittest discover -v tests

For examples of how to use the library, refer to :ref:`Usage Examples<examples>`.

Upgrading the Library
---------------------

To upgrade to the latest version of the library, run ``pip install -U textwiser``. If installing from source, you can
``git pull origin master`` in the repo folder to pull the latest stable commit, and follow the installation instructions
above.

Proxy Setup
-----------

In order to install the requirements and download pretrained models, a proxy setup may be required. Replace the proxy settings below with your own proxy configuration.

Anaconda
^^^^^^^^

Update your ``.condarc`` file to include the following lines:

.. code-block:: bash

    proxy_servers:
        http: http://<proxy_url>:<proxy_port>
        https: http://<proxy_url>:<proxy_port>


Pip
^^^

Use ``pip install --proxy http://<proxy_url>:<proxy_port> -r requirements.txt`` while installing the packages.

Unix Command Line
^^^^^^^^^^^^^^^^^

Add the following lines to your ``~/.bashrc``:

.. code-block:: bash

    export http_proxy=http://<proxy_url>:<proxy_port>
    export HTTPS_PROXY=$http_proxy
    export https_proxy=$http_proxy
    export HTTP_PROXY=$http_proxy
    export ALL_PROXY=$http_proxy # (required by cURL)


PyCharm
^^^^^^^

Add the following environment variables to your run configuration:

.. code-block:: bash

    HTTP_PROXY=http://<proxy_url>:<proxy_port>;HTTPS_PROXY=http://<proxy_url>:<proxy_port>
