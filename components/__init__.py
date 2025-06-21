"""
Netflix A/B Test Components Package

This package contains all the core components for running Bayesian A/B tests
on Netflix content recommendations with enhanced AI capabilities.

Components:
- enhanced_recommender: Advanced recommendation engine with SentenceTransformers + Anthropic
- bayesian_ab_test: Comprehensive Bayesian A/B testing framework
- data_handler: Netflix content data processing and management
- visualizations: Interactive charts and graphs for results analysis
"""

__version__ = "2.0.0"
__author__ = "Your Name"

# Import main classes for easy access
try:
    from .enhanced_recommender import EnhancedRecommendationEngine, MoodExtractor
    from .bayesian_ab_test import BayesianABTestEngine
    from .data_handler import NetflixDataHandler
    from .visualizations import ABTestVisualizer
    
    __all__ = [
        'EnhancedRecommendationEngine',
        'MoodExtractor', 
        'BayesianABTestEngine',
        'NetflixDataHandler',
        'ABTestVisualizer'
    ]
except ImportError as e:
    print(f"Warning: Could not import some components: {e}")
    __all__ = []