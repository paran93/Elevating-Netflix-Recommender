import numpy as np
import pandas as pd
from scipy.stats import beta
from typing import Dict, List, Tuple
import random
from .data_handler import NetflixDataHandler

class BayesianNetflixABTest:
    """Bayesian A/B test engine for Streamlit"""
    
    def __init__(self, netflix_data: Dict, priors: Dict):
        self.netflix_data = netflix_data
        self.priors = priors
        self.results = {
            'control': {'successes': 0, 'trials': 0},
            'treatment': {'successes': 0, 'trials': 0}
        }
        self.detailed_results = {'control': [], 'treatment': []}
        
    def run_full_test(self, num_users: int) -> Dict:
        """Run complete A/B test and return results"""
        
        # Generate user assignments
        assignments = ['control' if random.random() < 0.5 else 'treatment' 
                      for _ in range(num_users)]
        
        # Simulate user experiences
        for i, group in enumerate(assignments):
            # Generate realistic user query (simplified for Streamlit)
            user_query = self._generate_simple_query()
            
            if group == 'control':
                success = self._simulate_control_experience()
            else:
                success = self._simulate_treatment_experience()
            
            # Record result
            self.results[group]['trials'] += 1
            if success:
                self.results[group]['successes'] += 1
            
            # Store detailed result
            self.detailed_results[group].append({
                'user_id': i,
                'success': success,
                'query': user_query
            })
        
        return self._calculate_bayesian_summary()
    
    def _generate_simple_query(self) -> str:
        """Generate simple user query for demo"""
        queries = [
            "Looking for something funny to watch",
            "Need a good thriller",
            "What's good for family night?",
            "Want something romantic",
            "Looking for action movies",
            "Need comedy series to binge"
        ]
        return random.choice(queries)
    
    def _simulate_control_experience(self) -> bool:
        """Simulate control group experience"""
        # Low conversion rate for form redirect
        return np.random.random() < 0.05
    
    def _simulate_treatment_experience(self) -> bool:
        """Simulate treatment group experience"""
        # Higher conversion rate for AI recommendations
        return np.random.random() < 0.25
    
    def _calculate_bayesian_summary(self) -> Dict:
        """Calculate comprehensive Bayesian summary"""
        
        summary = {
            'sample_sizes': {
                'control': self.results['control']['trials'],
                'treatment': self.results['treatment']['trials']
            },
            'observed_rates': {
                'control': self.results['control']['successes'] / max(1, self.results['control']['trials']),
                'treatment': self.results['treatment']['successes'] / max(1, self.results['treatment']['trials'])
            },
            'posterior_stats': {
                'control': self._get_posterior_stats('control'),
                'treatment': self._get_posterior_stats('treatment')
            },
            'comparison': {
                'prob_treatment_better': self._calculate_prob_treatment_better(),
                'expected_lift': self._calculate_expected_lift()
            },
            'detailed_results': self.detailed_results
        }
        
        return summary
    
    def _get_posterior_stats(self, group: str) -> Dict:
        """Get posterior statistics for a group"""
        
        prior = self.priors[group]
        data = self.results[group]
        
        alpha_post = prior['alpha'] + data['successes']
        beta_post = prior['beta'] + (data['trials'] - data['successes'])
        
        posterior = beta(alpha_post, beta_post)
        
        return {
            'mean': alpha_post / (alpha_post + beta_post),
            'alpha': alpha_post,
            'beta': beta_post,
            'credible_interval_95': posterior.interval(0.95),
            'credible_interval_90': posterior.interval(0.90),
            'posterior_distribution': posterior
        }
    
    def _calculate_prob_treatment_better(self) -> float:
        """Calculate P(Treatment > Control)"""
        
        control_stats = self._get_posterior_stats('control')
        treatment_stats = self._get_posterior_stats('treatment')
        
        # Monte Carlo sampling
        n_samples = 100000
        control_samples = np.random.beta(control_stats['alpha'], control_stats['beta'], n_samples)
        treatment_samples = np.random.beta(treatment_stats['alpha'], treatment_stats['beta'], n_samples)
        
        return np.mean(treatment_samples > control_samples)
    
    def _calculate_expected_lift(self) -> Dict:
        """Calculate expected lift with credible intervals"""
        
        control_stats = self._get_posterior_stats('control')
        treatment_stats = self._get_posterior_stats('treatment')
        
        n_samples = 100000
        control_samples = np.random.beta(control_stats['alpha'], control_stats['beta'], n_samples)
        treatment_samples = np.random.beta(treatment_stats['alpha'], treatment_stats['beta'], n_samples)
        
        lift_samples = (treatment_samples - control_samples) / control_samples
        
        return {
            'expected_lift': np.mean(lift_samples),
            'lift_95_ci': np.percentile(lift_samples, [2.5, 97.5]),
            'probability_positive_lift': np.mean(lift_samples > 0)
        }