"""
Utility functions for the Netflix A/B Test application

This module provides helper functions for data loading, processing,
and common operations used throughout the application.
"""

__version__ = "1.0.0"

from .helpers import (
    setup_logging,
    load_json_data,
    load_text_queries,
    format_percentage,
    format_number,
    safe_divide,
    truncate_text
)

__all__ = [
    'setup_logging',
    'load_json_data',
    'load_text_queries',
    'format_percentage',
    'format_number',
    'safe_divide',
    'truncate_text'
]