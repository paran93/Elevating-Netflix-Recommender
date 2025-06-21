"""Utility functions for the Netflix A/B test application"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st
import pandas as pd
import numpy as np

def setup_logging(level: str = 'INFO'):
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def load_json_data(file_path: Path) -> Optional[Dict]:
    """Load JSON data from file with error handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading {file_path}: {e}")
        return None

def load_text_queries(file_path: Path) -> List[str]:
    """Load queries from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Parse queries (handle different formats)
            if content.startswith('[') or content.startswith('{'):
                # JSON format
                queries = json.loads(content)
                if isinstance(queries, dict):
                    queries = queries.get('queries', [])
            else:
                # Comma-separated format
                queries = [q.strip().strip("'\"") for q in content.split(',') if q.strip()]
            
            # Filter out short or invalid queries
            valid_queries = [q for q in queries if isinstance(q, str) and len(q.strip()) > 5]
            return valid_queries
            
    except Exception as e:
        st.error(f"❌ Error loading queries from {file_path}: {e}")
        return []

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format float as percentage"""
    return f"{value:.{decimals}%}"

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals"""
    return f"{value:.{decimals}f}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value for zero denominator"""
    return numerator / denominator if denominator != 0 else default

def truncate_text(text: str, max_length: int = 150) -> str:
    """Truncate text to specified length"""
    if not isinstance(text, str):
        text = str(text)
    
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(' ', 1)[0] + "..."

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        return False
    return True

def clean_numeric_column(series: pd.Series, default_value: float = 0.0) -> pd.Series:
    """Clean numeric column by handling NaN and invalid values"""
    return pd.to_numeric(series, errors='coerce').fillna(default_value)

def normalize_list_column(series: pd.Series) -> pd.Series:
    """Normalize a column that should contain lists"""
    def ensure_list(x):
        if pd.isna(x):
            return []
        elif isinstance(x, list):
            return [str(item) for item in x if item is not None]
        elif isinstance(x, str):
            return [x] if x.strip() else []
        else:
            return [str(x)] if x else []
    
    return series.apply(ensure_list)

def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    """Calculate confidence interval for a dataset"""
    if not data:
        return (0.0, 0.0)
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    std_error = np.std(data_array) / np.sqrt(len(data_array))
    
    # Use t-distribution for small samples
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    margin_error = t_value * std_error
    
    return (mean - margin_error, mean + margin_error)

def format_duration(minutes: float) -> str:
    """Format duration in minutes to human readable format"""
    if minutes < 60:
        return f"{minutes:.0f} min"
    elif minutes < 1440:  # Less than 24 hours
        hours = minutes / 60
        return f"{hours:.1f} hours"
    else:
        days = minutes / 1440
        return f"{days:.1f} days"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_computation_cache(func_name: str, *args, **kwargs):
    """Generic cache decorator for expensive computations"""
    # This is a placeholder - actual implementation would depend on the function
    pass

def log_user_action(action: str, details: Dict[str, Any] = None):
    """Log user actions for analytics"""
    logger = logging.getLogger(__name__)
    log_data = {
        'action': action,
        'timestamp': pd.Timestamp.now(),
        'details': details or {}
    }
    logger.info(f"User action: {json.dumps(log_data, default=str)}")

def get_color_palette(n_colors: int = 10) -> List[str]:
    """Get a color palette for visualizations"""
    netflix_colors = [
        '#e50914',  # Netflix red
        '#221f1f',  # Netflix black
        '#f5f5f1',  # Netflix white
        '#564d4d',  # Netflix gray
        '#0071eb',  # Bright blue
        '#00b4d8',  # Light blue
        '#90e0ef',  # Lighter blue
        '#ff006e',  # Hot pink
        '#fb8500',  # Orange
        '#ffb703'   # Yellow
    ]
    
    if n_colors <= len(netflix_colors):
        return netflix_colors[:n_colors]
    else:
        # Extend with standard plotly colors if needed
        import plotly.colors as pc
        return netflix_colors + pc.qualitative.Set3[:n_colors - len(netflix_colors)]