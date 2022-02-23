#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2019 FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifer: Apache-2.0

"""
:Author: FMR LLC
:Email: opensource@fidelity.com
:Version: 1.3.2 of Feb 23, 2022

This module defines the public interface of the
**TextWiser Library** providing access to the following modules:

    - ``TextWiser``
    - ``Embedding``
    - ``Transformation``
    - ``WordOptions``
    - ``PoolOptions``
    - ``device``
"""

__author__ = "FMR LLC"
__email__ = "opensource@fidelity.com"
__copyright__ = "Copyright (C) 2019, FMR LLC"

from textwiser.options import Embedding, PoolOptions, Transformation, WordOptions
from textwiser.textwiser import TextWiser
from textwiser.utils import device
from textwiser._version import __version__
