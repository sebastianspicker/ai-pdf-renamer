"""Compatibility facade for Folionym's public configuration API.

The implementation lives in :mod:`folionym.settings`; this module preserves
the documented import paths used by integrations and existing callers.
"""

from __future__ import annotations

from .settings.models import (
    ExtractionConfig,
    HeuristicCategoryConfig,
    HeuristicConfig,
    HeuristicScoreConfig,
    HeuristicSkipConfig,
    HeuristicWindowConfig,
    LLMBackendConfig,
    LLMConfig,
    LLMContentConfig,
    LLMRuntimeConfig,
    LLMVisionConfig,
    OutputConfig,
    OutputHookConfig,
    OutputModeConfig,
    OutputNamingConfig,
    OutputPathConfig,
    OutputProgressConfig,
    OutputTemplateConfig,
    OutputTraversalConfig,
    RenamerConfig,
    build_config_from_flat_dict,
)
from .settings.resolution import build_config

__all__ = [
    "ExtractionConfig",
    "HeuristicCategoryConfig",
    "HeuristicConfig",
    "HeuristicScoreConfig",
    "HeuristicSkipConfig",
    "HeuristicWindowConfig",
    "LLMBackendConfig",
    "LLMConfig",
    "LLMContentConfig",
    "LLMRuntimeConfig",
    "LLMVisionConfig",
    "OutputConfig",
    "OutputHookConfig",
    "OutputModeConfig",
    "OutputNamingConfig",
    "OutputPathConfig",
    "OutputProgressConfig",
    "OutputTemplateConfig",
    "OutputTraversalConfig",
    "RenamerConfig",
    "build_config",
    "build_config_from_flat_dict",
]
