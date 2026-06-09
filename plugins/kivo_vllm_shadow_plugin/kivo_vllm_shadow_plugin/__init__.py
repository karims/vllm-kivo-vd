# SPDX-License-Identifier: Apache-2.0

"""Kivo-VD vLLM shadow plugin and discovery utilities."""

from .internal_discovery import discover_internal_hooks
from .plugin import KivoShadowPluginState, register

__all__ = [
    "KivoShadowPluginState",
    "discover_internal_hooks",
    "register",
]
