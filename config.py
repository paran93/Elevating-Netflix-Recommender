"""Application configuration settings"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data files
NETFLIX_DATA_FILE = BASE_DIR / "netflix_content.json"
QUERIES_DATA_FILE = BASE_DIR / "list_queries.txt"

# API Configuration
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model settings
SENTENCE_TRANSFORMER_MODEL = os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
EMBEDDING_CACHE_DIR = BASE_DIR / '.cache' / 'embeddings'
EMBEDDING_CACHE_SIZE = int(os.getenv('EMBEDDING_CACHE_SIZE', '1000'))

# A/B Test Configuration
DEFAULT_TEST_SIZE = int(os.getenv('DEFAULT_TEST_SIZE', '500'))
MIN_SAMPLE_SIZE = 50
MAX_SAMPLE_SIZE = 2000

# Recommendation settings
CONFIDENCE_THRESHOLD = float(os.getenv('DEFAULT_CONFIDENCE_THRESHOLD', '0.3'))
MIN_RECOMMENDATIONS = int(os.getenv('DEFAULT_MIN_RECOMMENDATIONS', '3'))
MAX_RECOMMENDATIONS = 10

# Bayesian priors
BAYESIAN_PRIORS = {
    'control': {'alpha': 1, 'beta': 10},     # Pessimistic about form
    'treatment': {'alpha': 5, 'beta': 5}     # Neutral about AI
}

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# UI Configuration
APP_TITLE = "Netflix AI Recommendations A/B Test"
APP_ICON = "🎬"
SIDEBAR_STATE = "expanded"

# Performance settings
CACHE_TTL = 3600  # 1 hour
MAX_WORKERS = 4