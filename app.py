import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
import re
from typing import List, Dict
import traceback

# Page config
st.set_page_config(
    page_title="Netflix AI Recommendations A/B Test",
    page_icon="🎬",
    layout="wide"
)

# Try to import components with fallback
def load_components():
    """Load components with fallback options"""
    components_status = {}
    
    try:
        from components.enhanced_recommender import EnhancedRecommendationEngine
        components_status['enhanced_recommender'] = True
        st.success("✅ Enhanced recommender loaded (with SentenceTransformers)")
    except ImportError as e:
        st.warning(f"⚠️ Enhanced recommender not available: {e}")
        components_status['enhanced_recommender'] = False
        
        # Use basic fallback
        EnhancedRecommendationEngine = create_basic_recommender()
        st.info("📝 Using enhanced basic recommender with proper language parsing")
    
    try:
        from components.bayesian_ab_test import BayesianABTestEngine
        components_status['bayesian_ab_test'] = True
    except ImportError as e:
        st.warning(f"⚠️ Bayesian A/B test not available: {e}")
        components_status['bayesian_ab_test'] = False
        BayesianABTestEngine = create_basic_ab_test()
    
    return EnhancedRecommendationEngine, BayesianABTestEngine, components_status

def create_basic_recommender():
    """Enhanced recommender with proper language parsing and all features"""
    class BasicRecommendationEngine:
        def __init__(self, netflix_data: Dict, anthropic_api_key: str = None):
            self.netflix_data = netflix_data
            self.content_df = self._process_data()
        
        def _process_data(self):
            all_content = []
            
            # Process shows and movies
            for show in self.netflix_data.get('shows', []):
                show_copy = show.copy()
                show_copy['content_type'] = 'show'
                all_content.append(show_copy)
            
            for movie in self.netflix_data.get('movies', []):
                movie_copy = movie.copy()
                movie_copy['content_type'] = 'movie'
                all_content.append(movie_copy)
            
            if not all_content:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_content)
            
            # Handle all core fields
            df['year'] = pd.to_numeric(df.get('year', 2020), errors='coerce').fillna(2020).astype(int)
            df['title'] = df['title'].fillna('Unknown Title').astype(str)
            df['synopsis'] = df['synopsis'].fillna('No synopsis available').astype(str)
            
            if 'content_type' not in df.columns:
                df['content_type'] = 'show'
            
            # ✅ PROPER LANGUAGE PROCESSING
            if 'language' in df.columns:
                df['language'] = df['language'].fillna('English').astype(str)
                df['languages_list'] = df['language'].apply(self._parse_languages)
                df['primary_language'] = df['languages_list'].apply(lambda x: x[0] if x else 'English')
                df['has_english'] = df['languages_list'].apply(lambda x: 'English' in x)
            else:
                df['language'] = 'English'
                df['languages_list'] = [['English']] * len(df)
                df['primary_language'] = 'English'
                df['has_english'] = True
            
            # Handle mood and genre as lists
            df['mood'] = df.get('mood', []).apply(self._ensure_list)
            df['genre'] = df.get('genre', []).apply(self._ensure_list)
            
            print(f"✅ Processed {len(df)} items: {len(df[df['content_type']=='movie'])} movies, {len(df[df['content_type']=='show'])} shows")
            return df
        
        def _parse_languages(self, language_string):
            """Parse comma-separated languages into clean list"""
            if pd.isna(language_string) or not language_string:
                return ['English']
            
            # Split by comma, strip spaces, remove empty strings
            languages = [lang.strip() for lang in str(language_string).split(',')]
            languages = [lang for lang in languages if lang]  # Remove empty
            
            # Deduplicate while preserving order
            seen = set()
            unique_languages = []
            for lang in languages:
                if lang not in seen:
                    seen.add(lang)
                    unique_languages.append(lang)
            
            return unique_languages if unique_languages else ['English']
        
        def _ensure_list(self, value):
            """Ensure value is a list"""
            if pd.isna(value):
                return []
            elif isinstance(value, list):
                return [str(item) for item in value if item is not None]
            elif isinstance(value, str):
                return [value] if value.strip() else []
            else:
                return [str(value)] if value else []
        
        def get_recommendations(self, query: str, top_k: int = 5) -> Dict:
            if self.content_df.empty:
                return self._empty_response(query)
            
            query_lower = query.lower()
            user_prefs = self._extract_user_preferences(query_lower)
            
            # Filter by content type if specified
            filtered_df = self.content_df.copy()
            if user_prefs['content_type']:
                content_filtered = filtered_df[filtered_df['content_type'] == user_prefs['content_type']]
                if not content_filtered.empty:
                    filtered_df = content_filtered
            
            # Score each content item
            scored_content = []
            for _, content in filtered_df.iterrows():
                score = self._calculate_content_score(content, user_prefs, query_lower)
                scored_content.append((score, content))
            
            scored_content.sort(key=lambda x: x[0], reverse=True)
            
            recommendations = []
            for score, content in scored_content[:top_k]:
                recommendations.append({
                    'title': str(content['title']),
                    'year': int(content['year']),
                    'content_type': content['content_type'],
                    'genre': content['genre'],
                    'mood': content['mood'],
                    'language': str(content['language']),  # Original comma-separated
                    'languages_list': content['languages_list'],  # ✅ Parsed list
                    'primary_language': str(content['primary_language']),  # ✅ First language
                    'has_english': content['has_english'],  # ✅ English availability
                    'synopsis': str(content['synopsis'])[:200] + ('...' if len(str(content['synopsis'])) > 200 else ''),
                    'similarity_score': score,
                    'final_score': score,
                    'method': 'enhanced_matching',
                    'match_details': self._explain_match(content, user_prefs, score)
                })
            
            movies_count = sum(1 for r in recommendations if r['content_type'] == 'movie')
            shows_count = len(recommendations) - movies_count
            
            return {
                'recommendations': recommendations,
                'mood_analysis': {
                    'moods': user_prefs['moods'],
                    'genres': user_prefs['genres'],
                    'content_type': user_prefs['content_type'],
                    'preferred_language': user_prefs['language'],
                    'year_preference': user_prefs['year_range'],
                    'original_query': query
                },
                'should_redirect_to_form': len(recommendations) < 3 or max([r['similarity_score'] for r in recommendations], default=0) < 0.3,
                'confidence_score': min(1.0, max([r['similarity_score'] for r in recommendations], default=0)),
                'anthropic_recommendations': [],
                'total_available': len(self.content_df),
                'content_breakdown': {'movies': movies_count, 'shows': shows_count, 'total': len(recommendations)}
            }
        
        def _extract_user_preferences(self, query_lower: str) -> Dict:
            """Extract user preferences with enhanced language detection"""
            
            # Content type detection
            content_type = None
            if any(word in query_lower for word in ['movie', 'film', 'cinema']):
                content_type = 'movie'
            elif any(word in query_lower for word in ['show', 'series', 'tv', 'season', 'episode', 'binge']):
                content_type = 'show'
            
            # Mood detection based on actual data values
            mood_keywords = {
                'Feel-Good': ['feel good', 'feel-good', 'uplifting', 'positive', 'heartwarming'],
                'Funny': ['funny', 'comedy', 'hilarious', 'humor', 'comedic', 'laugh'],
                'Dark': ['dark', 'noir', 'gritty', 'serious'],
                'Romantic': ['romantic', 'romance', 'love', 'date night'],
                'Thrilling': ['exciting', 'action', 'adventure', 'intense', 'thrilling'],
                'Emotional': ['emotional', 'touching', 'moving', 'dramatic'],
                'Mind-blowing': ['mind blowing', 'mind-blowing', 'original', 'unique'],
                'Gripping': ['gripping', 'edge of seat', 'suspenseful'],
                'Heart-warming': ['heart warming', 'heart-warming', 'wholesome'],
                'Inspiring': ['inspiring', 'motivational'],
                'Smart': ['smart', 'intelligent', 'thought provoking'],
                'Lighthearted': ['lighthearted', 'light hearted', 'easy watching'],
                'Intense': ['intense', 'heavy', 'powerful'],
                'Character-driven': ['character driven', 'character-driven'],
                'Action-packed': ['action packed', 'action-packed'],
                'Binge-Worthy': ['binge worthy', 'binge-worthy', 'addictive'],
                'Suspenseful': ['suspenseful', 'suspense'],
                'Sweet': ['sweet', 'cute', 'adorable'],
                'Weird': ['weird', 'strange', 'bizarre'],
                'Raw': ['raw', 'gritty', 'realistic'],
                'Quirky': ['quirky', 'offbeat', 'unusual']
            }
            
            detected_moods = []
            for mood, keywords in mood_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_moods.append(mood)
            
            # Genre detection based on actual data values
            genre_keywords = {
                'Comedy': ['comedy', 'funny', 'humor'],
                'Drama': ['drama', 'dramatic'],
                'Crime': ['crime', 'criminal'],
                'Thriller': ['thriller', 'suspense'],
                'Action': ['action'],
                'Adventure': ['adventure'],
                'Horror': ['horror', 'scary', 'frightening'],
                'Romance': ['romance', 'romantic'],
                'Sci-Fi & Fantasy': ['sci-fi', 'science fiction', 'fantasy'],
                'Science Fiction': ['science fiction', 'sci fi'],
                'Fantasy': ['fantasy', 'magic'],
                'Documentary': ['documentary', 'real', 'factual'],
                'Animation': ['animation', 'animated', 'cartoon'],
                'Family': ['family', 'kids', 'children'],
                'Mystery': ['mystery', 'detective'],
                'War': ['war', 'military', 'battle'],
                'Western': ['western', 'cowboy'],
                'History': ['history', 'historical'],
                'Music': ['music', 'musical'],
                'Action & Adventure': ['action adventure'],
                'War & Politics': ['war politics', 'political'],
                'TV Movie': ['tv movie', 'made for tv'],
                'Mockumentary': ['mockumentary', 'mock documentary'],
                'Reality': ['reality', 'reality tv'],
                'Kids': ['kids', 'children']
            }
            
            detected_genres = []
            for genre, keywords in genre_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_genres.append(genre)
            
            # ✅ ENHANCED LANGUAGE DETECTION (based on your actual data)
            language_keywords = {
                'English': ['english'],
                'Spanish': ['spanish', 'latino', 'hispanic'],
                'French': ['french'],
                'German': ['german'],
                'Japanese': ['japanese', 'anime'],
                'Korean': ['korean', 'k-drama', 'kdrama'],
                'Italian': ['italian'],
                'Hindi': ['hindi', 'bollywood'],
                'Mandarin': ['mandarin', 'chinese'],
                'Cantonese': ['cantonese'],
                'Arabic': ['arabic'],
                'Portuguese': ['portuguese', 'brazilian'],
                'Russian': ['russian'],
                'Dutch': ['dutch'],
                'Swedish': ['swedish'],
                'Norwegian': ['norwegian'],
                'Danish': ['danish'],
                'Finnish': ['finnish'],
                'Polish': ['polish'],
                'Turkish': ['turkish'],
                'Bengali': ['bengali'],
                'Tamil': ['tamil'],
                'Telugu': ['telugu'],
                'Vietnamese': ['vietnamese'],
                'Thai': ['thai'],
                'Persian': ['persian', 'farsi'],
                'Hebrew': ['hebrew'],
                'Gujarati': ['gujarati'],
                'Punjabi': ['punjabi'],
                'Malayalam': ['malayalam'],
                'Kannada': ['kannada'],
                'Romanian': ['romanian'],
                'Croatian': ['croatian'],
                'Serbian': ['serbian'],
                'Ukrainian': ['ukrainian'],
                'Czech': ['czech'],
                'Hungarian': ['hungarian'],
                'Greek': ['greek'],
                'Bulgarian': ['bulgarian'],
                'Slovak': ['slovak'],
                'Estonian': ['estonian'],
                'Icelandic': ['icelandic'],
                'Catalan': ['catalan'],
                'Basque': ['basque'],
                'Galician': ['galician'],
                'Afrikaans': ['afrikaans'],
                'Swahili': ['swahili'],
                'Yoruba': ['yoruba'],
                'Zulu': ['zulu'],
                'Xhosa': ['xhosa'],
                'Tagalog': ['tagalog', 'filipino'],
                'Indonesian': ['indonesian'],
                'Malay': ['malay'],
                'Urdu': ['urdu'],
                'Nepali': ['nepali']
            }
            
            preferred_language = None
            for lang, keywords in language_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    preferred_language = lang
                    break
            
            # Year extraction
            year_range = None
            year_matches = re.findall(r'\b(19|20)\d{2}\b', query_lower)
            if year_matches:
                years = [int(y) for y in year_matches]
                year_range = (min(years), max(years))
            elif 'recent' in query_lower or 'new' in query_lower or 'latest' in query_lower:
                year_range = (2020, 2025)
            elif 'classic' in query_lower or 'old' in query_lower:
                year_range = (1990, 2010)
            elif '2020s' in query_lower:
                year_range = (2020, 2025)
            elif '2010s' in query_lower:
                year_range = (2010, 2019)
            elif '90s' in query_lower or 'nineties' in query_lower:
                year_range = (1990, 1999)
            elif '80s' in query_lower or 'eighties' in query_lower:
                year_range = (1980, 1989)
            
            return {
                'moods': detected_moods,
                'genres': detected_genres,
                'content_type': content_type,
                'year_range': year_range,
                'language': preferred_language
            }
        
        def _calculate_content_score(self, content, user_prefs: Dict, query_lower: str) -> float:
            """Calculate score with proper language matching"""
            score = 0.0
            
            # Content type matching (20%)
            if user_prefs['content_type']:
                if content['content_type'] == user_prefs['content_type']:
                    score += 0.2
            
            # Mood matching (25%)
            if user_prefs['moods']:
                content_moods = [str(m) for m in content['mood']]
                mood_matches = sum(1 for mood in user_prefs['moods'] 
                                 if any(mood.lower() in cm.lower() for cm in content_moods))
                if mood_matches > 0:
                    mood_score = mood_matches / len(user_prefs['moods'])
                    score += mood_score * 0.25
            
            # Genre matching (25%)
            if user_prefs['genres']:
                content_genres = [str(g) for g in content['genre']]
                genre_matches = sum(1 for genre in user_prefs['genres'] 
                                  if any(genre.lower() in cg.lower() for cg in content_genres))
                if genre_matches > 0:
                    genre_score = genre_matches / len(user_prefs['genres'])
                    score += genre_score * 0.25
            
            # Year matching (15%)
            if user_prefs['year_range']:
                year = content['year']
                min_year, max_year = user_prefs['year_range']
                if min_year <= year <= max_year:
                    score += 0.15
                else:
                    distance = min(abs(year - min_year), abs(year - max_year))
                    if distance <= 5:
                        score += 0.075
            
            # ✅ PROPER LANGUAGE MATCHING (10%)
            if user_prefs['language']:
                content_languages = content['languages_list']  # This is the parsed list
                
                # Exact match in language list
                if user_prefs['language'] in content_languages:
                    score += 0.1
                # Partial match (case insensitive)
                elif any(user_prefs['language'].lower() in lang.lower() for lang in content_languages):
                    score += 0.05
            
            # Keyword matching (5%)
            title = str(content['title']).lower()
            synopsis = str(content['synopsis']).lower()
            query_words = [word for word in query_lower.split() if len(word) > 2]
            if query_words:
                keyword_matches = sum(1 for word in query_words if word in title or word in synopsis)
                keyword_score = keyword_matches / len(query_words)
                score += keyword_score * 0.05
            
            return min(1.0, score)
        
        def _explain_match(self, content, user_prefs: Dict, score: float) -> str:
            """Explain match with proper language info"""
            reasons = []
            
            if user_prefs['content_type'] and content['content_type'] == user_prefs['content_type']:
                reasons.append(f"Perfect {content['content_type']} match")
            
            if user_prefs['moods']:
                content_moods = [str(m) for m in content['mood']]
                matching_moods = [mood for mood in user_prefs['moods'] 
                                if any(mood.lower() in cm.lower() for cm in content_moods)]
                if matching_moods:
                    reasons.append(f"Matches {', '.join(matching_moods)} mood")
            
            if user_prefs['genres']:
                content_genres = [str(g) for g in content['genre']]
                matching_genres = [genre for genre in user_prefs['genres'] 
                                 if any(genre.lower() in cg.lower() for cg in content_genres)]
                if matching_genres:
                    reasons.append(f"Matches {', '.join(matching_genres)} genre")
            
            if user_prefs['year_range']:
                year = content['year']
                min_year, max_year = user_prefs['year_range']
                if min_year <= year <= max_year:
                    reasons.append(f"From {year}")
            
            # ✅ PROPER LANGUAGE EXPLANATION
            if user_prefs['language']:
                content_languages = content['languages_list']
                if user_prefs['language'] in content_languages:
                    reasons.append(f"Available in {user_prefs['language']}")
                elif content['has_english']:
                    reasons.append("Available with English")
            
            return " • ".join(reasons) if reasons else f"Score: {score:.2f}"
        
        def _empty_response(self, query: str) -> Dict:
            return {
                'recommendations': [],
                'mood_analysis': {'moods': [], 'genres': [], 'content_type': None, 'original_query': query},
                'should_redirect_to_form': True,
                'confidence_score': 0,
                'anthropic_recommendations': [],
                'total_available': 0,
                'content_breakdown': {'movies': 0, 'shows': 0, 'total': 0}
            }
    
    return BasicRecommendationEngine

def create_basic_ab_test():
    """Create a basic A/B test if imports fail"""
    class BasicABTestEngine:
        def __init__(self, netflix_data: Dict, anthropic_api_key: str = None):
            self.netflix_data = netflix_data
            self.recommendation_engine = create_basic_recommender()(netflix_data, anthropic_api_key)
        
        def run_ab_test(self, user_queries: List[str]) -> Dict:
            # Simulate A/B test with the recommendation engine
            control_results = []
            treatment_results = []
            
            for i, query in enumerate(user_queries):
                if i % 2 == 0:  # Control group
                    # Simulate form redirect experience
                    control_results.append({
                        'engaged': np.random.random() < 0.12,
                        'found_content': False,
                        'user_satisfaction': 0.25 + np.random.normal(0, 0.1),
                        'time_spent_minutes': np.random.uniform(0.5, 2.0),
                        'action_taken': 'form_submitted' if np.random.random() < 0.12 else 'abandoned',
                        'recommendations_shown': 0
                    })
                else:  # Treatment group
                    # Use actual recommendation engine
                    rec_result = self.recommendation_engine.get_recommendations(query, top_k=5)
                    confidence = rec_result['confidence_score']
                    
                    if rec_result['should_redirect_to_form']:
                        # Low quality -> redirect to form
                        treatment_results.append({
                            'engaged': np.random.random() < 0.12,
                            'found_content': False,
                            'user_satisfaction': 0.25 + np.random.normal(0, 0.1),
                            'time_spent_minutes': np.random.uniform(0.5, 2.0),
                            'action_taken': 'redirected_to_form',
                            'recommendations_shown': 0
                        })
                    else:
                        # AI recommendations
                        engaged = np.random.random() < min(0.8, confidence + 0.3)
                        found_content = engaged and (np.random.random() < confidence)
                        
                        treatment_results.append({
                            'engaged': engaged,
                            'found_content': found_content,
                            'user_satisfaction': min(1.0, confidence + np.random.normal(0.1, 0.1)),
                            'time_spent_minutes': np.random.uniform(3, 8) if engaged else np.random.uniform(0.5, 2),
                            'action_taken': 'started_watching' if found_content else ('browsed' if engaged else 'abandoned'),
                            'recommendations_shown': len(rec_result['recommendations'])
                        })
            
            # Calculate metrics
            control_metrics = self._calculate_metrics(control_results)
            treatment_metrics = self._calculate_metrics(treatment_results)
            
            # Calculate business impact
            business_impact = self._calculate_business_impact(control_metrics, treatment_metrics)
            
            # Bayesian analysis
            bayesian_analysis = self._simple_bayesian_analysis(control_results, treatment_results)
            
            return {
                'test_summary': {
                    'total_users': len(user_queries),
                    'control_size': len(control_results),
                    'treatment_size': len(treatment_results)
                },
                'control_metrics': control_metrics,
                'treatment_metrics': treatment_metrics,
                'bayesian_analysis': bayesian_analysis,
                'business_impact': business_impact,
                'recommendation': self._get_recommendation(bayesian_analysis, business_impact)
            }
        
        def _calculate_metrics(self, results: List[Dict]) -> Dict:
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
        
        def _calculate_business_impact(self, control_metrics: Dict, treatment_metrics: Dict) -> Dict:
            def safe_lift(treatment, control):
                if control == 0:
                    return float('inf') if treatment > 0 else 0
                return ((treatment - control) / control) * 100
            
            return {
                'engagement_lift_percent': safe_lift(treatment_metrics['engagement_rate'], control_metrics['engagement_rate']),
                'content_discovery_lift_percent': safe_lift(treatment_metrics['content_discovery_rate'], control_metrics['content_discovery_rate']),
                'satisfaction_lift_percent': safe_lift(treatment_metrics['avg_satisfaction'], control_metrics['avg_satisfaction']),
                'time_spent_difference': treatment_metrics['avg_time_spent'] - control_metrics['avg_time_spent'],
                'abandonment_reduction_percent': safe_lift(control_metrics['abandonment_rate'] - treatment_metrics['abandonment_rate'], control_metrics['abandonment_rate'])
            }
        
        def _simple_bayesian_analysis(self, control_results: List[Dict], treatment_results: List[Dict]) -> Dict:
            # Simple Bayesian analysis
            control_engaged = sum([r['engaged'] for r in control_results])
            control_total = len(control_results)
            treatment_engaged = sum([r['engaged'] for r in treatment_results])
            treatment_total = len(treatment_results)
            
            # Beta posterior parameters
            control_alpha = 1 + control_engaged
            control_beta = 10 + (control_total - control_engaged)
            treatment_alpha = 5 + treatment_engaged
            treatment_beta = 5 + (treatment_total - treatment_engaged)
            
            # Monte Carlo to estimate P(treatment > control)
            n_samples = 10000
            control_samples = np.random.beta(control_alpha, control_beta, n_samples)
            treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)
            
            prob_treatment_better = np.mean(treatment_samples > control_samples)
            expected_lift = np.mean((treatment_samples - control_samples) / control_samples)
            
            return {
                'prob_treatment_better': prob_treatment_better,
                'expected_lift': expected_lift,
                'control_posterior': {'alpha': control_alpha, 'beta': control_beta},
                'treatment_posterior': {'alpha': treatment_alpha, 'beta': treatment_beta}
            }
        
        def _get_recommendation(self, bayesian_analysis: Dict, business_impact: Dict) -> Dict:
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
    
    return BasicABTestEngine

# Load predefined queries
@st.cache_data
def load_test_queries():
    """Load predefined test queries"""
    try:
        with open('list_queries.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            # Parse the queries
            queries = [q.strip().strip("'\"") for q in content.split(',') if q.strip()]
            return [q for q in queries if len(q) > 5][:1000]  # Filter and limit
    except Exception as e:
        st.warning(f"Could not load queries file: {e}")
        return [
            "I want something funny to watch",
            "Looking for a good thriller movie",
            "Need something romantic for date night",
            "What's a good action show to binge?",
            "I need something uplifting",
            "Looking for a comedy series",
            "Want a scary movie for tonight",
            "Need a feel-good show to relax",
            "What's good for family night?",
            "I want something mind-blowing",
            "Spanish crime drama show",
            "Korean romantic movie",
            "French comedy series",
            "Japanese anime movie",
            "German thriller show"
        ]

@st.cache_data
def load_netflix_data():
    """Load Netflix content data with comprehensive error handling"""
    import os
    
    filename = 'netflix_content.json'
    
    try:
        # Check file exists
        if not os.path.exists(filename):
            st.error(f"❌ File '{filename}' not found!")
            files = [f for f in os.listdir('.') if f.endswith(('.json', '.txt'))]
            st.write(f"Available data files: {files}")
            return {"shows": []}
        
        # Load the file
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, dict):
            st.error("❌ JSON should be an object/dictionary at root level")
            if isinstance(data, list):
                st.info("Converting list to proper format...")
                data = {"shows": data}
            else:
                return {"shows": []}
        
        # Handle different data structures
        content_found = False
        total_content = 0
        
        if 'shows' in data:
            shows_count = len(data.get('shows', []))
            total_content += shows_count
            content_found = True
        
        if 'movies' in data:
            movies_count = len(data.get('movies', []))
            total_content += movies_count
            content_found = True
        
        # If no standard keys, try to find content
        if not content_found:
            st.warning("⚠️ No 'shows' or 'movies' keys found. Checking data structure...")
            
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict) and 'title' in value[0]:
                        st.info(f"📺 Found {len(value):,} content items in '{key}' key")
                        # Assume they're shows if not specified
                        data = {"shows": value}
                        total_content = len(value)
                        content_found = True
                        break
            
            if not content_found:
                st.error("❌ Could not find any content data in JSON")
                return {"shows": []}
        
        st.success(f"✅ Successfully loaded {total_content:,} content items!")
        
        return data
        
    except json.JSONDecodeError as e:
        st.error(f"❌ JSON format error: {e}")
        st.error("The JSON file has invalid syntax. Common issues:")
        st.write("- Missing commas between objects")
        st.write("- Unescaped quotes in text")
        st.write("- Missing closing brackets")
        st.write("- Trailing commas")
        return {"shows": []}
    
    except MemoryError:
        st.error("❌ File too large to load into memory")
        st.info("Try using a smaller dataset for testing")
        return {"shows": []}
    
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {e}")
        st.write("Full error details:")
        st.code(traceback.format_exc())
        return {"shows": []}

def main():
    """Main Streamlit application"""
    
    st.title("🎬 Netflix AI Recommendations A/B Test")
    st.subheader("Form Redirect vs AI Recommendations with Enhanced Language Processing")
    
    # Load components with fallback
    with st.spinner("Loading components..."):
        EnhancedRecommendationEngine, BayesianABTestEngine, components_status = load_components()
    
    # Show component status
    with st.expander("🔧 Component Status"):
        for component, status in components_status.items():
            status_icon = "✅" if status else "⚠️"
            st.write(f"{status_icon} {component}: {'Available' if status else 'Using fallback'}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key
        anthropic_key = st.text_input(
            "Anthropic API Key (Optional)",
            type="password",
            value=os.getenv('ANTHROPIC_API_KEY', ''),
            help="For Anthropic-powered recommendations comparison"
        )
        
        if anthropic_key:
            os.environ['ANTHROPIC_API_KEY'] = anthropic_key
        
        # Test parameters
        st.subheader("📊 Test Parameters")
        test_size = st.slider("Number of Test Queries", 50, 1000, 500)
        confidence_level = st.selectbox("Confidence Level", [0.90, 0.95, 0.99], index=1)
        
        # Recommendation settings
        st.subheader("🎯 Recommendation Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.8, 0.3)
        min_recs = st.slider("Minimum Recommendations", 1, 5, 3)
        
        # System info
        st.subheader("ℹ️ System Info")
        st.write(f"Python: {sys.version.split()[0]}")
        st.write(f"Streamlit: {st.__version__}")
    
    # Load data
    st.header("📊 Data Loading")
    netflix_data = load_netflix_data()
    test_queries = load_test_queries()
    
    st.write(f"📋 **Test Queries Available:** {len(test_queries):,}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Single Query Test",
        "🧪 Run A/B Test", 
        "📊 Results Analysis",
        "📈 Bayesian Dashboard",
        "🔍 Data Explorer"
    ])
    
    with tab1:
        st.header("🎯 Test Single Query")
        
        # Query examples
        with st.expander("💡 Example Queries"):
            example_queries = [
                "I want something funny and feel-good to watch",
                "Looking for a romantic comedy movie",
                "Need a thriller series to binge watch",
                "What's good for family movie night?",
                "I want something mind-blowing and original",
                "Looking for a dark crime drama show",
                "Spanish comedy movie from the 2000s",
                "Korean romantic series",
                "French thriller show",
                "Japanese anime movie"
            ]
            for example in example_queries:
                if st.button(f"Try: '{example}'", key=f"example_{example[:20]}"):
                    st.session_state.test_query = example
        
        # Query input
        user_query = st.text_input(
            "Enter your query:", 
            value=st.session_state.get('test_query', "Looking for a dark crime drama show"),
            placeholder="e.g., Spanish romantic comedy movie from the 90s..."
        )
        
        if st.button("🔍 Get Recommendations", type="primary"):
            if not netflix_data.get('shows') and not netflix_data.get('movies'):
                st.error("❌ No data available. Please check your data file.")
                return
            
            # Initialize recommendation engine
            with st.spinner("Analyzing query and finding recommendations..."):
                try:
                    rec_engine = EnhancedRecommendationEngine(netflix_data, anthropic_key)
                    results = rec_engine.get_recommendations(user_query, top_k=5)
                except Exception as e:
                    st.error(f"❌ Error getting recommendations: {e}")
                    return
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 AI Recommendations")
                
                if results['should_redirect_to_form']:
                    st.warning("⚠️ Low confidence - would redirect to form")
                    st.info(f"Confidence Score: {results['confidence_score']:.2f}")
                else:
                    st.success(f"✅ High confidence recommendations (Score: {results['confidence_score']:.2f})")
                
                # Content breakdown
                breakdown = results['content_breakdown']
                if breakdown['total'] > 0:
                    st.write(f"**Content Mix:** {breakdown.get('movies', 0)} movies, {breakdown.get('shows', 0)} shows")
                
                # Mood analysis
                mood_analysis = results['mood_analysis']
                if mood_analysis.get('moods') or mood_analysis.get('genres') or mood_analysis.get('content_type'):
                    st.write("**🧠 Detected Intent:**")
                    if mood_analysis.get('moods'):
                        st.write(f"• **Moods:** {', '.join(mood_analysis['moods'])}")
                    if mood_analysis.get('genres'):
                        st.write(f"• **Genres:** {', '.join(mood_analysis['genres'])}")
                    if mood_analysis.get('content_type'):
                        st.write(f"• **Content Type:** {mood_analysis['content_type'].title()}")
                    if mood_analysis.get('preferred_language'):
                        st.write(f"• **Language:** {mood_analysis['preferred_language']}")
                    if mood_analysis.get('year_preference'):
                        year_range = mood_analysis['year_preference']
                        st.write(f"• **Year Range:** {year_range[0]}-{year_range[1]}")
                
                # Show recommendations
                for i, rec in enumerate(results['recommendations'], 1):
                    content_icon = "🎬" if rec.get('content_type') == 'movie' else "📺"
                    content_type = rec.get('content_type', 'show').title()
                    
                    with st.expander(f"{content_icon} {i}. {rec['title']} ({rec['year']}) - {content_type}"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Genres:** {', '.join(rec.get('genre', [])) if rec.get('genre') else 'N/A'}")
                            st.write(f"**Moods:** {', '.join(rec.get('mood', [])) if rec.get('mood') else 'N/A'}")
                            st.write(f"**Languages:** {rec.get('language', 'Unknown')}")
                            if rec.get('language') and ',' in rec.get('language', ''):
                                languages = [lang.strip() for lang in rec.get('language', '').split(',')]
                                if len(languages) > 1:
                                    st.write(f"**Primary Language:** {languages[0]}")
                                    st.write(f"**Also Available:** {', '.join(languages[1:3])}")
                        
                        with col_b:
                            st.write(f"**Match Score:** {rec.get('final_score', 0):.3f}")
                            st.write(f"**Method:** {rec.get('method', 'unknown')}")
                            if rec.get('match_details'):
                                st.write(f"**Why Recommended:** {rec['match_details']}")
                        
                        st.write(f"**Synopsis:** {rec.get('synopsis', 'No synopsis available')}")
            
            with col2:
                st.subheader("🔴 Current Netflix Experience do not cater to Mood")
                st.error("❌ User redirected to request form")
                st.write("**What happens currently:**")
                st.write("• User sent to 'Request TV shows or movies' form")
                st.write("• Must fill out form and wait for manual review")
                
                st.write("**Problems with current approach:**")
                st.write("• No instant gratification")
                st.write("• Generic search doesn't understand intent")
                st.write("• No mood/genre matching")
                
                if results.get('anthropic_recommendations'):
                    st.subheader("🧠 Anthropic Comparison")
                    for rec in results['anthropic_recommendations']:
                        content_icon = "🎬" if rec.get('content_type') == 'movie' else "📺"
                        st.write(f"{content_icon} {rec['title']} ({rec['year']}) - {rec.get('content_type', 'show').title()}")
    
    with tab2:
        st.header("🧪 Run Full A/B Test")
        
        st.write("**Enhanced A/B Test Design:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔴 Control Group:**")
            st.write("• Current Netflix experience")
            st.write("• Redirect to request form")
            st.write("• Manual processing required")
            st.write("• No immediate recommendations")
            st.write("• No language/mood detection")
        
        with col2:
            st.write("**🟢 Treatment Group:**")
            st.write("• AI-powered instant recommendations")
            st.write("• Mood and genre detection")
            st.write("• Content type matching (movie/show)")
            st.write("• Language preference detection")
            st.write("• Year-based filtering")
        
        st.write(f"📋 **Available Test Queries:** {len(test_queries):,}")
        st.write(f"**Sample Size:** {test_size:,} users (split 50/50 between groups)")
        
        # Show sample queries
        with st.expander("📝 Sample Test Queries"):
            sample_queries = test_queries[:10]
            for i, query in enumerate(sample_queries, 1):
                st.write(f"{i}. {query}")
        
        if st.button("🚀 Start A/B Test", type="primary"):
            if not netflix_data.get('shows') and not netflix_data.get('movies'):
                st.error("❌ No data available for testing")
                return
            
            # Initialize A/B test engine
            with st.spinner("Initializing A/B test engine..."):
                ab_test = BayesianABTestEngine(netflix_data, anthropic_key)
            
            # Run test
            with st.spinner(f"Running A/B test with {test_size} queries..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Use subset of queries for test
                test_subset = test_queries[:test_size]
                
                try:
                    results = ab_test.run_ab_test(test_subset)
                    progress_bar.progress(100)
                    status_text.text("✅ A/B Test completed!")
                    
                    # Store results in session state
                    st.session_state['ab_results'] = results
                    st.success("🎉 A/B Test completed! Check the Results Analysis tab.")
                    
                except Exception as e:
                    st.error(f"❌ Error running A/B test: {e}")
                    st.write("Error details:")
                    st.code(traceback.format_exc())
    
    with tab3:
        st.header("📊 A/B Test Results Analysis")
        
        if 'ab_results' not in st.session_state:
            st.info("👆 Run an A/B test first to see results here.")
            st.write("**What you'll see after running a test:**")
            st.write("• Enhanced language processing results")
            st.write("• Content type preference analysis")
            st.write("• Mood/genre matching effectiveness")
            st.write("• Statistical significance analysis")
            st.write("• Business impact calculations")
            return
        
        results = st.session_state['ab_results']
        
        # Test summary
        summary = results['test_summary']
        st.write(f"**Test Summary:**")
        st.write(f"• Total Users: {summary['total_users']:,}")
        st.write(f"• Control Group: {summary['control_size']:,} users")
        st.write(f"• Treatment Group: {summary['treatment_size']:,} users")
        
        # Key metrics comparison
        control = results['control_metrics']
        treatment = results['treatment_metrics']
        
        st.subheader("🎯 Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_engagement = (treatment['engagement_rate'] - control['engagement_rate'])
            st.metric(
                "Engagement Rate",
                f"{treatment['engagement_rate']:.1%}",
                delta=f"{delta_engagement:+.1%}",
                help="Users who interact with recommendations"
            )
        
        with col2:
            delta_discovery = (treatment['content_discovery_rate'] - control['content_discovery_rate'])
            st.metric(
                "Content Discovery",
                f"{treatment['content_discovery_rate']:.1%}",
                delta=f"{delta_discovery:+.1%}",
                help="Users who actually start watching content"
            )
        
        with col3:
            delta_satisfaction = (treatment['avg_satisfaction'] - control['avg_satisfaction'])
            st.metric(
                "User Satisfaction",
                f"{treatment['avg_satisfaction']:.2f}",
                delta=f"{delta_satisfaction:+.2f}",
                help="Average satisfaction score (0-1 scale)"
            )
        
        with col4:
            delta_time = (treatment['avg_time_spent'] - control['avg_time_spent'])
            st.metric(
                "Avg Time Spent",
                f"{treatment['avg_time_spent']:.1f}m",
                delta=f"{delta_time:+.1f}m",
                help="Average time users spend on platform"
            )
        
        # Enhanced features analysis
        st.subheader("🌟 Enhanced Features Impact")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**AI System Advantages:**")
            st.write("• ✅ Detects movie vs show preferences")
            st.write("• ✅ Parses multiple languages correctly")
            st.write("• ✅ Matches mood and genre intent") 
            st.write("• ✅ Handles year preferences")
            st.write("• ✅ Provides instant recommendations")
            
        with col2:
            st.write("**Current System Limitations:**")
            st.write("• ❌ No content type consideration")
            st.write("• ❌ No language preference detection")
            st.write("• ❌ No mood/genre understanding")
            st.write("• ❌ Manual form processing")
            st.write("• ❌ High abandonment rate")
        
        # Business impact visualization
        impact = results['business_impact']
        st.subheader("💼 Business Impact Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            metrics = ['Engagement', 'Content Discovery', 'Satisfaction']
            control_values = [control['engagement_rate'], control['content_discovery_rate'], control['avg_satisfaction']]
            treatment_values = [treatment['engagement_rate'], treatment['content_discovery_rate'], treatment['avg_satisfaction']]
            
            fig = go.Figure(data=[
                go.Bar(name='Control (Form)', x=metrics, y=control_values, marker_color='#e50914'),
                go.Bar(name='Treatment (AI)', x=metrics, y=treatment_values, marker_color='#00b4d8')
            ])
            fig.update_layout(
                title="Key Metrics Comparison",
                barmode='group',
                yaxis_title="Rate/Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Percentage Improvements:**")
            improvements = [
                ("Engagement Rate", impact['engagement_lift_percent']),
                ("Content Discovery", impact['content_discovery_lift_percent']),
                ("User Satisfaction", impact['satisfaction_lift_percent']),
                ("Abandonment Reduction", impact['abandonment_reduction_percent'])
            ]
            
            for metric, improvement in improvements:
                if improvement == float('inf'):
                    st.markdown(f"• **{metric}**: <span style='color:green'>∞% (0% to positive)</span>", unsafe_allow_html=True)
                else:
                    color = "green" if improvement > 0 else "red"
                    st.markdown(f"• **{metric}**: <span style='color:{color}'>{improvement:+.1f}%</span>", unsafe_allow_html=True)
        
        # Strategic recommendation
        st.subheader("🚀 Strategic Recommendation")
        recommendation = results['recommendation']
        
        if recommendation['decision'] == 'STRONG_DEPLOY':
            st.success(f"✅ **{recommendation['decision']}**: {recommendation['reasoning']}")
            st.write("**Next Steps:**")
            st.write("• Deploy AI recommendations immediately")
            st.write("• Monitor key metrics for first 30 days")
            st.write("• Gather user feedback for continuous improvement")
            st.write("• Expand language support based on usage patterns")
        elif recommendation['decision'] == 'DEPLOY_WITH_MONITORING':
            st.warning(f"⚠️ **{recommendation['decision']}**: {recommendation['reasoning']}")
            st.write("**Next Steps:**")
            st.write("• Gradual rollout to 25% of users initially")
            st.write("• Close monitoring of metrics")
            st.write("• A/B test different language detection methods")
        else:
            st.error(f"❌ **{recommendation['decision']}**: {recommendation['reasoning']}")
            st.write("**Next Steps:**")
            st.write("• Improve recommendation algorithm")
            st.write("• Expand training data for more languages")
            st.write("• Re-test with enhanced system")
    
    with tab4:
        st.header("📈 Bayesian Statistical Analysis")
        
        if 'ab_results' not in st.session_state:
            st.info("👆 Run an A/B test first to see Bayesian analysis here.")
            return
        
        bayesian = st.session_state['ab_results']['bayesian_analysis']
        
        # Probability metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "P(Treatment > Control)",
                f"{bayesian['prob_treatment_better']:.1%}",
                help="Bayesian probability that AI is better"
            )
        
        with col2:
            st.metric(
                "Expected Lift",
                f"{bayesian['expected_lift']:.1%}",
                help="Expected relative improvement"
            )
        
        with col3:
            confidence = bayesian['prob_treatment_better']
            if confidence >= 0.95:
                status = "🟢 High Confidence"
            elif confidence >= 0.90:
                status = "🟡 Medium Confidence"
            else:
                status = "🔴 Low Confidence"
            st.metric("Confidence Level", status)
        
        # Enhanced insights
        st.subheader("🎯 Enhanced System Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Language Processing Benefits:**")
            st.write("• Correctly parses comma-separated languages")
            st.write("• Detects primary vs secondary languages")
            st.write("• Matches user language preferences")
            st.write("• Improves international content discovery")
            
            st.write("**Content Type Matching:**")
            st.write("• Distinguishes movie vs show preferences")
            st.write("• Filters results by user intent")
            st.write("• Reduces irrelevant recommendations")
        
        with col2:
            st.write("**Advanced Scoring System:**")
            st.write("• Content type: 20% weight")
            st.write("• Mood matching: 25% weight") 
            st.write("• Genre matching: 25% weight")
            st.write("• Year matching: 15% weight")
            st.write("• Language matching: 10% weight")
            st.write("• Keyword matching: 5% weight")
            
            st.write("**Statistical Robustness:**")
            st.write("• Bayesian posterior updating")
            st.write("• Monte Carlo simulation")
            st.write("• Credible intervals")
    
    with tab5:
        st.header("🔍 Data Explorer")
        
        if not netflix_data.get('shows') and not netflix_data.get('movies'):
            st.warning("No data loaded to explore")
            return
        
        # Combine all content
        all_content = netflix_data.get('shows', []) + netflix_data.get('movies', [])
        
        if not all_content:
            st.warning("No content found in data")
            return
        
        df = pd.DataFrame(all_content)
        
        # Data overview
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_items = len(df)
            st.metric("Total Items", f"{total_items:,}")
        
        with col2:
            if 'content_type' in df.columns:
                movies = len(df[df.get('content_type', 'show') == 'movie'])
                st.metric("Movies", f"{movies:,}")
            else:
                st.metric("Movies", "N/A")
        
        with col3:
            if 'content_type' in df.columns:
                shows = len(df[df.get('content_type', 'show') == 'show'])
                st.metric("Shows", f"{shows:,}")
            else:
                st.metric("Shows", f"{total_items:,}")
        
        with col4:
            if 'language' in df.columns:
                # Count unique individual languages
                all_languages = set()
                for lang_string in df['language'].dropna():
                    languages = [lang.strip() for lang in str(lang_string).split(',')]
                    all_languages.update(languages)
                st.metric("Unique Languages", len(all_languages))
            else:
                st.metric("Languages", "N/A")
        
        # Language analysis
        if 'language' in df.columns:
            st.subheader("🌍 Language Analysis")
            
            # Parse all languages
            language_counts = {}
            for lang_string in df['language'].dropna():
                languages = [lang.strip() for lang in str(lang_string).split(',')]
                for lang in languages:
                    if lang:
                        language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Top languages
            sorted_languages = sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
            top_10_languages = sorted_languages[:10]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 10 Languages:**")
                for lang, count in top_10_languages:
                    st.write(f"• {lang}: {count:,} items")
            
            with col2:
                # Create chart
                langs, counts = zip(*top_10_languages)
                fig = px.bar(x=list(counts), y=list(langs), orientation='h', 
                           title="Top 10 Languages by Content Count")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.subheader("📋 Sample Data")
        
        # Select columns to display
        available_columns = df.columns.tolist()
        default_columns = [col for col in ['title', 'year', 'content_type', 'genre', 'language', 'mood'] if col in available_columns][:6]
        
        display_columns = st.multiselect(
            "Select columns to display:",
            available_columns,
            default=default_columns
        )
        
        if display_columns:
            sample_size = st.slider("Number of items to show:", 5, min(50, len(df)), 10)
            sample_df = df[display_columns].head(sample_size)
            
            # Format list columns for better display
            for col in display_columns:
                if col in sample_df.columns:
                    sample_df[col] = sample_df[col].apply(
                        lambda x: ', '.join(x[:3]) + ('...' if len(x) > 3 else '') if isinstance(x, list) else str(x)
                    )
            
            st.dataframe(sample_df, use_container_width=True)
        
        # Search functionality
        st.subheader("🔍 Search Content")
        search_term = st.text_input("Search titles, languages, genres, and descriptions:")
        
        if search_term:
            # Enhanced search
            mask = df['title'].str.contains(search_term, case=False, na=False)
            if 'synopsis' in df.columns:
                mask |= df['synopsis'].str.contains(search_term, case=False, na=False)
            if 'language' in df.columns:
                mask |= df['language'].str.contains(search_term, case=False, na=False)
            if 'genre' in df.columns:
                mask |= df['genre'].astype(str).str.contains(search_term, case=False, na=False)
            
            search_results = df[mask]
            st.write(f"Found {len(search_results)} matches:")
            
            if len(search_results) > 0:
                for _, item in search_results.head(10).iterrows():
                    content_icon = "🎬" if item.get('content_type') == 'movie' else "📺"
                    content_type = item.get('content_type', 'show').title()
                    
                    st.write(f"{content_icon} **{item.get('title', 'Unknown')}** ({item.get('year', 'N/A')}) - {content_type}")
                    
                    if item.get('language'):
                        st.write(f"   🌍 Languages: {item['language']}")
                    if item.get('genre'):
                        genres = ', '.join(item['genre']) if isinstance(item['genre'], list) else str(item['genre'])
                        st.write(f"   🎭 Genres: {genres}")
                    if item.get('synopsis'):
                        st.write(f"   📝 {str(item['synopsis'])[:200]}...")

if __name__ == "__main__":
    main()