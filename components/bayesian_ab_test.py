import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta
import random
from typing import Dict, List, Tuple
from .enhanced_recommender import EnhancedRecommendationEngine

class BayesianABTestEngine:
    """Enhanced Bayesian A/B testing for recommendation systems"""
    
    def __init__(self, netflix_data: Dict, anthropic_api_key: str = None):
        self.netflix_data = netflix_data
        self.recommendation_engine = EnhancedRecommendationEngine(netflix_data, anthropic_api_key)
        
        # A/B test configuration
        self.test_groups = {
            'control': 'form_redirect',      # Current: redirect to form
            'treatment': 'ai_recommendations' # New: AI recommendations
        }
        
        # Prior beliefs for Bayesian analysis
        self.priors = {
            'control': {'alpha': 1, 'beta': 10},    # Pessimistic: form has low engagement
            'treatment': {'alpha': 5, 'beta': 5}     # Optimistic: AI might work better
        }
        
        self.test_results = {
            'control': [],
            'treatment': []
        }
    
    def run_ab_test(self, user_queries: List[str], test_size: int = None) -> Dict:
        """Run A/B test with user queries"""
        
        if test_size:
            user_queries = user_queries[:test_size]
        
        print(f"🧪 Starting A/B Test with {len(user_queries)} queries")
        
        # Randomly assign users to groups
        assignments = ['control' if random.random() < 0.5 else 'treatment' 
                      for _ in range(len(user_queries))]
        
        results = []
        
        for i, (query, group) in enumerate(zip(user_queries, assignments)):
            if i % 100 == 0:
                print(f"   Processing query {i+1}/{len(user_queries)}")
            
            if group == 'control':
                result = self._simulate_control_experience(query)
            else:
                result = self._simulate_treatment_experience(query)
            
            result.update({
                'query_id': i,
                'query': query,
                'group': group
            })
            
            results.append(result)
            self.test_results[group].append(result)
        
        print("✅ A/B Test completed")
        return self._analyze_results()
    
    def _simulate_control_experience(self, query: str) -> Dict:
        """Simulate current experience: redirect to form"""
        
        # Current Netflix experience: user gets redirected to form
        # Most users abandon, few complete form
        form_completion_rate = 0.12  # Only 12% complete the form
        user_satisfaction = 0.2      # Low satisfaction (doesn't solve immediate need)
        
        form_completed = random.random() < form_completion_rate
        found_content = False  # Form doesn't provide immediate content
        time_spent = random.uniform(1, 3) if form_completed else random.uniform(0.5, 1.5)
        
        return {
            'engaged': form_completed,
            'found_content': found_content,
            'user_satisfaction': user_satisfaction + np.random.normal(0, 0.1),  # FIXED: use np.random.normal
            'time_spent_minutes': time_spent,
            'action_taken': 'form_completed' if form_completed else 'abandoned',
            'recommendations_shown': 0
        }
    
    def _simulate_treatment_experience(self, query: str) -> Dict:
        """Simulate AI recommendation experience"""
        
        # Get AI recommendations
        rec_result = self.recommendation_engine.get_recommendations(query, top_k=5)
        recommendations = rec_result['recommendations']
        should_redirect = rec_result['should_redirect_to_form']
        confidence_score = rec_result['confidence_score']
        
        if should_redirect:
            # Low quality recommendations -> redirect to form
            return self._simulate_control_experience(query)
        
        # Simulate user behavior based on recommendation quality
        engagement_probability = min(0.8, confidence_score + 0.2)
        satisfaction_base = confidence_score
        
        engaged = random.random() < engagement_probability
        
        if engaged and recommendations:
            # User clicked on recommendations
            found_content = random.random() < (confidence_score * 0.9)  # Higher confidence = more likely to watch
            time_spent = random.uniform(3, 8)  # More time exploring recommendations
            satisfaction = min(1.0, satisfaction_base + np.random.normal(0.1, 0.1))  # FIXED: use np.random.normal
            action = 'started_watching' if found_content else 'browsed_recommendations'
        else:
            # User didn't engage
            found_content = False
            time_spent = random.uniform(0.5, 2)
            satisfaction = satisfaction_base + np.random.normal(0, 0.1)  # FIXED: use np.random.normal
            action = 'abandoned'
        
        return {
            'engaged': engaged,
            'found_content': found_content,
            'user_satisfaction': max(0, min(1, satisfaction)),
            'time_spent_minutes': time_spent,
            'action_taken': action,
            'recommendations_shown': len(recommendations),
            'confidence_score': confidence_score,
            'anthropic_available': len(rec_result['anthropic_recommendations']) > 0
        }
    
    def _analyze_results(self) -> Dict:
        """Perform Bayesian analysis of A/B test results"""
        
        control_results = self.test_results['control']
        treatment_results = self.test_results['treatment']
        
        # Calculate key metrics
        control_metrics = self._calculate_group_metrics(control_results)
        treatment_metrics = self._calculate_group_metrics(treatment_results)
        
        # Bayesian analysis
        bayesian_analysis = self._bayesian_comparison(control_results, treatment_results)
        
        # Business impact
        business_impact = self._calculate_business_impact(control_metrics, treatment_metrics)
        
        return {
            'test_summary': {
                'total_users': len(control_results) + len(treatment_results),
                'control_size': len(control_results),
                'treatment_size': len(treatment_results)
            },
            'control_metrics': control_metrics,
            'treatment_metrics': treatment_metrics,
            'bayesian_analysis': bayesian_analysis,
            'business_impact': business_impact,
            'recommendation': self._get_recommendation(bayesian_analysis, business_impact)
        }
    
    def _calculate_group_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics for a test group"""
        if not results:
            return {}
        
        return {
            'sample_size': len(results),
            'engagement_rate': np.mean([r['engaged'] for r in results]),
            'content_discovery_rate': np.mean([r['found_content'] for r in results]),
            'avg_satisfaction': np.mean([r['user_satisfaction'] for r in results]),
            'avg_time_spent': np.mean([r['time_spent_minutes'] for r in results]),
            'abandonment_rate': np.mean([r['action_taken'] == 'abandoned' for r in results]),
            'avg_recommendations_shown': np.mean([r.get('recommendations_shown', 0) for r in results])
        }
    
    def _bayesian_comparison(self, control_results: List[Dict], treatment_results: List[Dict]) -> Dict:
        """Perform Bayesian comparison between groups"""
        
        # Engagement rate comparison
        control_engaged = sum([r['engaged'] for r in control_results])
        control_total = len(control_results)
        treatment_engaged = sum([r['engaged'] for r in treatment_results])
        treatment_total = len(treatment_results)
        
        # Update priors with observed data
        control_posterior = {
            'alpha': self.priors['control']['alpha'] + control_engaged,
            'beta': self.priors['control']['beta'] + (control_total - control_engaged)
        }
        
        treatment_posterior = {
            'alpha': self.priors['treatment']['alpha'] + treatment_engaged,
            'beta': self.priors['treatment']['beta'] + (treatment_total - treatment_engaged)
        }
        
        # Monte Carlo simulation for comparison
        n_simulations = 100000
        control_samples = np.random.beta(control_posterior['alpha'], control_posterior['beta'], n_simulations)
        treatment_samples = np.random.beta(treatment_posterior['alpha'], treatment_posterior['beta'], n_simulations)
        
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # Expected lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        expected_lift = np.mean(lift_samples)
        lift_credible_interval = np.percentile(lift_samples, [2.5, 97.5])
        
        return {
            'control_posterior': control_posterior,
            'treatment_posterior': treatment_posterior,
            'prob_treatment_better': prob_treatment_better,
            'expected_lift': expected_lift,
            'lift_credible_interval': lift_credible_interval.tolist(),
            'control_engagement_rate': control_engaged / control_total,
            'treatment_engagement_rate': treatment_engaged / treatment_total
        }
    
    def _calculate_business_impact(self, control_metrics: Dict, treatment_metrics: Dict) -> Dict:
        """Calculate business impact metrics"""
        
        # Helper function for safe division
        def safe_lift_calculation(treatment_val, control_val):
            if control_val == 0:
                return float('inf') if treatment_val > 0 else 0
            return ((treatment_val - control_val) / control_val) * 100
        
        return {
            'engagement_lift_percent': safe_lift_calculation(
                treatment_metrics['engagement_rate'], 
                control_metrics['engagement_rate']
            ),
            'content_discovery_lift_percent': safe_lift_calculation(
                treatment_metrics['content_discovery_rate'], 
                control_metrics['content_discovery_rate']
            ),
            'satisfaction_lift_percent': safe_lift_calculation(
                treatment_metrics['avg_satisfaction'], 
                control_metrics['avg_satisfaction']
            ),
            'time_spent_difference': treatment_metrics['avg_time_spent'] - control_metrics['avg_time_spent'],
            'abandonment_reduction_percent': safe_lift_calculation(
                control_metrics['abandonment_rate'] - treatment_metrics['abandonment_rate'], 
                control_metrics['abandonment_rate']
            )
        }
    
    def _get_recommendation(self, bayesian_analysis: Dict, business_impact: Dict) -> Dict:
        """Generate recommendation based on results"""
        
        prob_better = bayesian_analysis['prob_treatment_better']
        expected_lift = bayesian_analysis['expected_lift']
        
        if prob_better >= 0.95 and expected_lift > 0.1:
            decision = "STRONG_DEPLOY"
            reasoning = f"High confidence ({prob_better:.1%}) with strong lift ({expected_lift:.1%})"
        elif prob_better >= 0.90 and expected_lift > 0.05:
            decision = "DEPLOY_WITH_MONITORING"
            reasoning = f"Good confidence ({prob_better:.1%}) with positive lift ({expected_lift:.1%})"
        elif prob_better >= 0.80:
            decision = "GRADUAL_ROLLOUT"
            reasoning = f"Moderate confidence ({prob_better:.1%}), consider gradual rollout"
        else:
            decision = "DO_NOT_DEPLOY"
            reasoning = f"Insufficient evidence ({prob_better:.1%})"
        
        return {
            'decision': decision,
            'reasoning': reasoning,
            'confidence_level': prob_better
        }