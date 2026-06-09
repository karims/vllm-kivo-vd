# SPDX-License-Identifier: Apache-2.0

"""Kivo-VD vLLM general-plugin feasibility probe."""

from .plugin import KivoShadowPluginState, register

__all__ = ["KivoShadowPluginState", "register"]
