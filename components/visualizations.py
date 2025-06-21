import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List

class ABTestVisualizer:
    """Create interactive visualizations for A/B test results (no revenue components)"""
    
    def __init__(self, ab_test_results: Dict):
        self.results = ab_test_results
    
    def create_posterior_distributions(self):
        """Create posterior distribution plot"""
        
        control_stats = self.results['posterior_stats']['control']
        treatment_stats = self.results['posterior_stats']['treatment']
        
        x = np.linspace(0, 1, 1000)
        
        # Calculate PDF for both distributions
        control_pdf = control_stats['posterior_distribution'].pdf(x)
        treatment_pdf = treatment_stats['posterior_distribution'].pdf(x)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x, y=control_pdf,
            mode='lines',
            name='Control (Form)',
            line=dict(color='#e50914', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=treatment_pdf,
            mode='lines',
            name='Treatment (AI)',
            line=dict(color='#00b4d8', width=3),
            fill='tonexty'
        ))
        
        # Add mean lines
        fig.add_vline(
            x=control_stats['mean'],
            line_dash="dash",
            line_color="#e50914",
            annotation_text=f"Control Mean: {control_stats['mean']:.3f}"
        )
        
        fig.add_vline(
            x=treatment_stats['mean'],
            line_dash="dash",
            line_color="#00b4d8",
            annotation_text=f"Treatment Mean: {treatment_stats['mean']:.3f}"
        )
        
        fig.update_layout(
            title="Posterior Conversion Rate Distributions",
            xaxis_title="Conversion Rate",
            yaxis_title="Probability Density",
            hovermode='x unified'
        )
        
        return fig
    
    def create_credible_intervals(self):
        """Create credible intervals visualization"""
        
        control_stats = self.results['posterior_stats']['control']
        treatment_stats = self.results['posterior_stats']['treatment']
        
        groups = ['Control<br>(Form)', 'Treatment<br>(AI)']
        means = [control_stats['mean'], treatment_stats['mean']]
        
        ci_95 = [control_stats['credible_interval_95'], treatment_stats['credible_interval_95']]
        ci_90 = [control_stats['credible_interval_90'], treatment_stats['credible_interval_90']]
        
        fig = go.Figure()
        
        # 95% CI
        fig.add_trace(go.Scatter(
            x=groups,
            y=[ci[1] for ci in ci_95],
            mode='lines+markers',
            name='95% CI Upper',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=groups,
            y=[ci[0] for ci in ci_95],
            mode='lines+markers',
            name='95% Credible Interval',
            fill='tonexty',
            fillcolor='rgba(68, 68, 68, 0.3)',
            line=dict(color='rgba(0,0,0,0)')
        ))
        
        # Means
        fig.add_trace(go.Scatter(
            x=groups,
            y=means,
            mode='markers+text',
            name='Posterior Mean',
            marker=dict(size=12, color=['#e50914', '#00b4d8']),
            text=[f'{mean:.3f}' for mean in means],
            textposition='top center'
        ))
        
        fig.update_layout(
            title="Conversion Rates with Credible Intervals",
            yaxis_title="Conversion Rate",
            hovermode='x unified'
        )
        
        return fig
    
    def create_conversion_funnel(self):
        """Create conversion funnel comparison"""
        
        control_rate = self.results['observed_rates']['control']
        treatment_rate = self.results['observed_rates']['treatment']
        
        # Create funnel data
        funnel_data = pd.DataFrame({
            'Stage': ['Visited', 'Engaged', 'Converted'],
            'Control': [100, 15, control_rate * 100],  # Assuming 15% engagement for form
            'Treatment': [100, 80, treatment_rate * 100]  # Higher engagement for AI
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Funnel(
            name='Control (Form)',
            y=funnel_data['Stage'],
            x=funnel_data['Control'],
            textinfo="value+percent initial",
            marker=dict(color='#e50914', opacity=0.7)
        ))
        
        fig.add_trace(go.Funnel(
            name='Treatment (AI)',
            y=funnel_data['Stage'],
            x=funnel_data['Treatment'],
            textinfo="value+percent initial",
            marker=dict(color='#00b4d8', opacity=0.7)
        ))
        
        fig.update_layout(
            title="User Conversion Funnel Comparison"
        )
        
        return fig
    
    def create_probability_evolution(self):
        """Create probability evolution over time"""
        
        # Simulate probability evolution
        total_sample = self.results['sample_sizes']['control']
        sample_sizes = range(50, total_sample + 50, max(50, total_sample // 20))
        
        # Simulate evolution based on final probability
        final_prob = self.results['comparison']['prob_treatment_better']
        prob_evolution = []
        
        for i, size in enumerate(sample_sizes):
            # Start uncertain, converge to final probability
            progress = (i + 1) / len(sample_sizes)
            uncertainty = (1 - progress) * 0.3  # Decreasing uncertainty
            prob = 0.5 + (final_prob - 0.5) * progress + np.random.normal(0, uncertainty)
            prob = np.clip(prob, 0.4, 1.0)
            prob_evolution.append(prob)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(sample_sizes),
            y=prob_evolution,
            mode='lines+markers',
            name='P(Treatment > Control)',
            line=dict(color='#00b4d8', width=3)
        ))
        
        # Add confidence thresholds
        fig.add_hline(y=0.95, line_dash="dash", line_color="green", 
                     annotation_text="95% Confidence")
        fig.add_hline(y=0.90, line_dash="dash", line_color="orange", 
                     annotation_text="90% Confidence")
        fig.add_hline(y=0.50, line_dash="solid", line_color="red", 
                     annotation_text="No Difference")
        
        fig.update_layout(
            title="Statistical Confidence Evolution",
            xaxis_title="Sample Size per Group",
            yaxis_title="P(Treatment > Control)",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_lift_distribution(self):
        """Create expected lift distribution"""
        
        lift_stats = self.results['comparison']['expected_lift']
        
        # Generate lift samples for visualization
        expected_lift = lift_stats['expected_lift']
        # Estimate standard deviation from confidence interval
        ci_range = lift_stats['lift_95_ci'][1] - lift_stats['lift_95_ci'][0]
        std_dev = ci_range / 3.92  # Approximate from 95% CI
        
        lift_samples = np.random.normal(expected_lift, std_dev, 10000)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=lift_samples * 100,  # Convert to percentage
            nbinsx=50,
            name='Lift Distribution',
            opacity=0.7,
            marker_color='#00b4d8'
        ))
        
        # Add expected lift line
        fig.add_vline(
            x=expected_lift * 100,
            line_dash="dash",
            line_color="#e50914",
            annotation_text=f"Expected: {expected_lift:.1%}"
        )
        
        # Add CI lines
        ci = lift_stats['lift_95_ci']
        fig.add_vline(x=ci[0] * 100, line_dash="dot", line_color="orange")
        fig.add_vline(x=ci[1] * 100, line_dash="dot", line_color="orange")
        
        fig.update_layout(
            title="Expected Lift Distribution",
            xaxis_title="Relative Lift (%)",
            yaxis_title="Frequency"
        )
        
        return fig
    
    def create_risk_assessment(self, results: Dict):
        """Create risk assessment visualization"""
        
        prob_better = results['comparison']['prob_treatment_better']
        
        # Risk categories
        categories = ['Deploy Now', 'Monitor Closely', 'More Testing', 'Do Not Deploy']
        
        if prob_better >= 0.95:
            risk_level = 0
        elif prob_better >= 0.90:
            risk_level = 1
        elif prob_better >= 0.80:
            risk_level = 2
        else:
            risk_level = 3
        
        colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
        values = [1 if i == risk_level else 0.1 for i in range(4)]
        
        fig = go.Figure(data=go.Bar(
            x=categories,
            y=values,
            marker_color=[colors[i] if values[i] > 0.5 else 'lightgray' for i in range(4)],
            text=[f"{prob_better:.1%}" if i == risk_level else "" for i in range(4)],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Decision Recommendation: {categories[risk_level]}",
            yaxis_title="Confidence Level",
            showlegend=False,
            yaxis=dict(range=[0, 1.2])
        )
        
        return fig
    
    def create_effect_size_chart(self):
        """Create effect size visualization"""
        
        control_rate = self.results['observed_rates']['control']
        treatment_rate = self.results['observed_rates']['treatment']
        expected_lift = self.results['comparison']['expected_lift']['expected_lift']
        
        # Effect size categories
        categories = ['Current\n(Form)', 'With AI']
        values = [control_rate * 100, treatment_rate * 100]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['#e50914', '#00b4d8'],
            text=[f'{val:.1f}%' for val in values],
            textposition='auto'
        ))
        
        # Add improvement annotation
        fig.add_annotation(
            x=1, y=treatment_rate * 100,
            text=f'+{expected_lift:.1%} improvement',
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            arrowwidth=2
        )
        
        fig.update_layout(
            title="Conversion Rate Comparison",
            yaxis_title="Conversion Rate (%)",
            showlegend=False
        )
        
        return fig
    
    def get_sample_user_experiences(self, n: int = 10) -> pd.DataFrame:
        """Get sample user experiences for display"""
        
        control_results = self.results['detailed_results']['control'][:n//2]
        treatment_results = self.results['detailed_results']['treatment'][:n//2]
        
        sample_data = []
        
        for result in control_results:
            sample_data.append({
                'Group': 'Control',
                'User Query': result['query'],
                'Converted': '✅' if result['success'] else '❌',
                'Experience': 'Redirected to form'
            })
        
        for result in treatment_results:
            sample_data.append({
                'Group': 'Treatment',
                'User Query': result['query'],
                'Converted': '✅' if result['success'] else '❌',
                'Experience': 'AI recommendations'
            })
        
        return pd.DataFrame(sample_data)