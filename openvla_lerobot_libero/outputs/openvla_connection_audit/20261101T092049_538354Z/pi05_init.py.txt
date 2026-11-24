#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .configuration_pi05 import PI05Config


__all__ = ["PI05Config", "PI05Policy", "make_pi05_pre_post_processors"]


_LAZY_MODELS = {'PI05Policy': ('.modeling_pi05', 'PI05Policy')}

def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_MODELS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_MODELS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

_LAZY_MODELS["make_pi05_pre_post_processors"] = (
    ".processor_pi05",
    "make_pi05_pre_post_processors",
)
