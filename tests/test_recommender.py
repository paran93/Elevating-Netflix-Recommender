"""Unit tests for the enhanced recommendation engine"""
import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from components.enhanced_recommender import MoodExtractor, EnhancedRecommendationEngine

class TestMoodExtractor(unittest.TestCase):
    """Test cases for mood extraction functionality"""
    
    def setUp(self):
        self.extractor = MoodExtractor()
    
    def test_mood_extraction_funny(self):
        """Test extraction of funny mood"""
        query = "I want something funny to watch"
        result = self.extractor.extract_mood_and_genre(query)
        
        self.assertIn('funny', result['moods'])
        self.assertEqual(result['confidence'], 'high')
    
    def test_mood_extraction_romantic(self):
        """Test extraction of romantic mood"""
        query = "Looking for a romantic movie for date night"
        result = self.extractor.extract_mood_and_genre(query)
        
        self.assertIn('romantic', result['moods'])
        self.assertTrue(result['has_specific_request'])
    
    def test_genre_extraction(self):
        """Test extraction of genres"""
        query = "Looking for a good thriller or action movie"
        result = self.extractor.extract_mood_and_genre(query)
        
        self.assertIn('thriller', result['genres'])
        self.assertIn('action', result['genres'])
    
    def test_combined_mood_genre(self):
        """Test extraction of both mood and genre"""
        query = "I need a funny comedy show to binge"
        result = self.extractor.extract_mood_and_genre(query)
        
        self.assertIn('funny', result['moods'])
        self.assertIn('comedy', result['genres'])
    
    def test_low_confidence_query(self):
        """Test detection of low confidence queries"""
        query = "maybe something good"
        result = self.extractor.extract_mood_and_genre(query)
        
        self.assertEqual(result['confidence'], 'low')
    
    def test_empty_query(self):
        """Test handling of empty or minimal queries"""
        query = ""
        result = self.extractor.extract_mood_and_genre(query)
        
        self.assertEqual(result['moods'], [])
        self.assertEqual(result['genres'], [])

class TestEnhancedRecommendationEngine(unittest.TestCase):
    """Test cases for the enhanced recommendation engine"""
    
    def setUp(self):
        # Create sample Netflix data for testing
        self.sample_data = {
            "shows": [
                {
                    "title": "Test Comedy Show",
                    "year": 2023,
                    "genre": ["Comedy"],
                    "mood": ["Funny", "Feel-Good"],
                    "synopsis": "A hilarious comedy show about friends",
                    "language": "English"
                },
                {
                    "title": "Test Thriller",
                    "year": 2022,
                    "genre": ["Thriller", "Crime"],
                    "mood": ["Dark", "Exciting"],
                    "synopsis": "A gripping thriller about mystery",
                    "language": "English"
                },
                {
                    "title": "Test Romance",
                    "year": 2021,
                    "genre": ["Romance", "Drama"],
                    "mood": ["Romantic", "Emotional"],
                    "synopsis": "A beautiful love story",
                    "language": "Spanish"
                }
            ]
        }
        
        # Initialize without Anthropic API for testing
        self.engine = EnhancedRecommendationEngine(self.sample_data, anthropic_api_key=None)
    
    def test_initialization(self):
        """Test proper initialization of recommendation engine"""
        self.assertIsNotNone(self.engine.content_df)
        self.assertEqual(len(self.engine.content_df), 3)
        self.assertIsNotNone(self.engine.mood_extractor)
    
    def test_get_recommendations_comedy(self):
        """Test recommendations for comedy query"""
        query = "I want something funny to watch"
        results = self.engine.get_recommendations(query, top_k=2)
        
        self.assertIsNotNone(results['recommendations'])
        self.assertTrue(len(results['recommendations']) >= 1)
        
        # Should prioritize comedy content
        top_rec = results['recommendations'][0]
        self.assertIn('Comedy', top_rec.get('genre', []))
    
    def test_get_recommendations_thriller(self):
        """Test recommendations for thriller query"""
        query = "Looking for an exciting thriller"
        results = self.engine.get_recommendations(query, top_k=2)
        
        self.assertIsNotNone(results['recommendations'])
        top_rec = results['recommendations'][0]
        self.assertTrue(
            'Thriller' in top_rec.get('genre', []) or 
            'Exciting' in top_rec.get('mood', [])
        )
    
    def test_mood_analysis(self):
        """Test mood analysis in recommendation results"""
        query = "I need something romantic and emotional"
        results = self.engine.get_recommendations(query)
        
        mood_analysis = results['mood_analysis']
        self.assertIn('romantic', mood_analysis['moods'])
        self.assertIn('emotional', mood_analysis['moods'])
    
    def test_confidence_scoring(self):
        """Test confidence scoring mechanism"""
        # High confidence query
        high_conf_query = "I want a funny comedy show"
        results = self.engine.get_recommendations(high_conf_query)
        high_confidence = results['confidence_score']
        
        # Low confidence query
        low_conf_query = "something"
        results = self.engine.get_recommendations(low_conf_query)
        low_confidence = results['confidence_score']
        
        self.assertGreater(high_confidence, low_confidence)
    
    def test_form_redirection_logic(self):
        """Test when system should redirect to form"""
        # Very generic query should potentially redirect
        generic_query = "show me something"
        results = self.engine.get_recommendations(generic_query)
        
        # Should have redirection decision
        self.assertIn('should_redirect_to_form', results)
        self.assertIsInstance(results['should_redirect_to_form'], bool)
    
    def test_empty_query_handling(self):
        """Test handling of empty queries"""
        empty_query = ""
        results = self.engine.get_recommendations(empty_query)
        
        self.assertIsNotNone(results)
        self.assertIn('recommendations', results)
        self.assertIn('confidence_score', results)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)