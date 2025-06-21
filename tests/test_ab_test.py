"""Unit tests for Bayesian A/B testing engine"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from components.bayesian_ab_test import BayesianABTestEngine

class TestBayesianABTestEngine(unittest.TestCase):
    """Test cases for Bayesian A/B testing functionality"""
    
    def setUp(self):
        # Sample Netflix data for testing
        self.sample_data = {
            "shows": [
                {
                    "title": "Comedy Show",
                    "year": 2023,
                    "genre": ["Comedy"],
                    "mood": ["Funny"],
                    "synopsis": "A funny show",
                    "language": "English"
                },
                {
                    "title": "Drama Show", 
                    "year": 2022,
                    "genre": ["Drama"],
                    "mood": ["Emotional"],
                    "synopsis": "An emotional drama",
                    "language": "English"
                }
            ]
        }
        
        # Initialize A/B test engine without Anthropic for testing
        self.ab_test = BayesianABTestEngine(self.sample_data, anthropic_api_key=None)
    
    def test_initialization(self):
        """Test proper initialization of A/B test engine"""
        self.assertIsNotNone(self.ab_test.recommendation_engine)
        self.assertIn('control', self.ab_test.test_groups)
        self.assertIn('treatment', self.ab_test.test_groups)
        self.assertIn('control', self.ab_test.priors)
        self.assertIn('treatment', self.ab_test.priors)
    
    def test_control_simulation(self):
        """Test control group experience simulation"""
        query = "I want something funny"
        result = self.ab_test._simulate_control_experience(query)
        
        # Check required fields
        required_fields = ['engaged', 'found_content', 'user_satisfaction', 
                          'time_spent_minutes', 'action_taken', 'recommendations_shown']
        for field in required_fields:
            self.assertIn(field, result)
        
        # Control should not show recommendations
        self.assertEqual(result['recommendations_shown'], 0)
        # Control should not find content (redirected to form)
        self.assertFalse(result['found_content'])
    
    def test_treatment_simulation(self):
        """Test treatment group experience simulation"""
        query = "I want something funny"
        result = self.ab_test._simulate_treatment_experience(query)
        
        # Check required fields
        required_fields = ['engaged', 'found_content', 'user_satisfaction', 
                          'time_spent_minutes', 'action_taken']
        for field in required_fields:
            self.assertIn(field, result)
        
        # Treatment might show recommendations
        self.assertIn('recommendations_shown', result)
    
    def test_small_ab_test(self):
        """Test running a small A/B test"""
        test_queries = [
            "I want something funny",
            "Looking for a drama", 
            "Need something exciting",
            "What's good to watch?"
        ]
        
        results = self.ab_test.run_ab_test(test_queries)
        
        # Check result structure
        self.assertIn('test_summary', results)
        self.assertIn('control_metrics', results)
        self.assertIn('treatment_metrics', results)
        self.assertIn('bayesian_analysis', results)
        self.assertIn('business_impact', results)
        self.assertIn('recommendation', results)
    
    def test_group_metrics_calculation(self):
        """Test calculation of group metrics"""
        # Sample results
        sample_results = [
            {
                'engaged': True,
                'found_content': True,
                'user_satisfaction': 0.8,
                'time_spent_minutes': 5.0,
                'action_taken': 'started_watching',
                'recommendations_shown': 3
            },
            {
                'engaged': False,
                'found_content': False,
                'user_satisfaction': 0.3,
                'time_spent_minutes': 1.0,
                'action_taken': 'abandoned',
                'recommendations_shown': 0
            }
        ]
        
        metrics = self.ab_test._calculate_group_metrics(sample_results)
        
        # Check calculated metrics
        self.assertEqual(metrics['sample_size'], 2)
        self.assertEqual(metrics['engagement_rate'], 0.5)  # 1 out of 2 engaged
        self.assertEqual(metrics['content_discovery_rate'], 0.5)  # 1 out of 2 found content
        self.assertAlmostEqual(metrics['avg_satisfaction'], 0.55)  # (0.8 + 0.3) / 2
    
    def test_bayesian_comparison(self):
        """Test Bayesian comparison between groups"""
        # Mock some results
        control_results = [{'engaged': False} for _ in range(10)]  # All failed
        treatment_results = [{'engaged': True} for _ in range(8)] + [{'engaged': False} for _ in range(2)]  # 8/10 success
        
        comparison = self.ab_test._bayesian_comparison(control_results, treatment_results)
        
        # Check comparison structure
        self.assertIn('prob_treatment_better', comparison)
        self.assertIn('expected_lift', comparison)
        self.assertIn('control_posterior', comparison)
        self.assertIn('treatment_posterior', comparison)
        
        # Treatment should be better with high probability
        self.assertGreater(comparison['prob_treatment_better'], 0.8)
        self.assertGreater(comparison['expected_lift'], 0)
    
    def test_business_impact_calculation(self):
        """Test business impact metrics calculation"""
        control_metrics = {
            'engagement_rate': 0.1,
            'content_discovery_rate': 0.05,
            'avg_satisfaction': 0.3,
            'avg_time_spent': 2.0,
            'abandonment_rate': 0.9
        }
        
        treatment_metrics = {
            'engagement_rate': 0.4,
            'content_discovery_rate': 0.3,
            'avg_satisfaction': 0.7,
            'avg_time_spent': 5.0,
            'abandonment_rate': 0.6
        }
        
        impact = self.ab_test._calculate_business_impact(control_metrics, treatment_metrics)
        
        # Check impact calculations
        self.assertIn('engagement_lift_percent', impact)
        self.assertIn('content_discovery_lift_percent', impact)
        self.assertIn('satisfaction_lift_percent', impact)
        
        # All should be positive improvements
        self.assertGreater(impact['engagement_lift_percent'], 0)
        self.assertGreater(impact['content_discovery_lift_percent'], 0)
        self.assertGreater(impact['satisfaction_lift_percent'], 0)
    
    def test_recommendation_logic(self):
        """Test recommendation decision logic"""
        # High confidence scenario
        high_confidence = {
            'prob_treatment_better': 0.97,
            'expected_lift': 0.25
        }
        
        business_impact = {
            'engagement_lift_percent': 150
        }
        
        recommendation = self.ab_test._get_recommendation(high_confidence, business_impact)
        
        self.assertEqual(recommendation['decision'], 'STRONG_DEPLOY')
        self.assertGreater(recommendation['confidence_level'], 0.95)
        
        # Low confidence scenario
        low_confidence = {
            'prob_treatment_better': 0.65,
            'expected_lift': 0.05
        }
        
        recommendation = self.ab_test._get_recommendation(low_confidence, business_impact)
        self.assertIn(recommendation['decision'], ['GRADUAL_ROLLOUT', 'DO_NOT_DEPLOY'])

if __name__ == '__main__':
    unittest.main(verbosity=2)