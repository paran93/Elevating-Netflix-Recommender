import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import os
import re
from typing import Dict, List, Tuple, Optional
import json
import logging

class MoodExtractor:
    """Extract mood and intent from natural language queries"""
    
    def __init__(self):
        self.mood_patterns = {
            'funny': ['funny', 'comedy', 'hilarious', 'laugh', 'humor', 'amusing', 'witty', 'comedic'],
            'feel_good': ['feel good', 'uplifting', 'positive', 'cheerful', 'heartwarming', 'wholesome', 'happy'],
            'emotional': ['emotional', 'touching', 'moving', 'tear jerker', 'dramatic', 'deep', 'sad', 'cry'],
            'exciting': ['exciting', 'thrilling', 'action', 'adventure', 'intense', 'adrenaline', 'fast paced'],
            'romantic': ['romantic', 'love', 'romance', 'date night', 'relationship', 'couple'],
            'relaxing': ['relaxing', 'chill', 'easy watching', 'light', 'calm', 'peaceful', 'laid back'],
            'dark': ['dark', 'gritty', 'serious', 'noir', 'psychological', 'disturbing', 'heavy'],
            'inspiring': ['inspiring', 'motivational', 'uplifting', 'hope', 'triumph', 'overcoming'],
            'smart': ['smart', 'intelligent', 'thought provoking', 'cerebral', 'complex'],
            'suspenseful': ['suspenseful', 'gripping', 'edge of seat', 'tense', 'nail biting'],
            'depressing': ['depressing', 'sad', 'melancholy', 'tragic', 'heartbreaking'],
            'discussion_sparking': ['discussion sparking', 'thought provoking', 'controversial', 'debate'],
            'true_crime': ['true crime', 'true-crime', 'crime documentary', 'investigation']
        }
        
        self.genre_keywords = {
            'action': ['action', 'fight', 'chase', 'superhero', 'martial arts', 'action packed'],
            'comedy': ['comedy', 'funny', 'sitcom', 'humor', 'comedic'],
            'drama': ['drama', 'dramatic', 'serious', 'character driven'],
            'horror': ['horror', 'scary', 'frightening', 'spooky', 'creepy', 'terrifying'],
            'thriller': ['thriller', 'suspense', 'mystery', 'crime', 'detective'],
            'romance': ['romance', 'romantic', 'love story', 'relationship'],
            'sci_fi': ['sci-fi', 'science fiction', 'futuristic', 'space', 'alien', 'technology'],
            'documentary': ['documentary', 'real', 'true story', 'factual', 'educational'],
            'family': ['family', 'kids', 'children', 'all ages', 'wholesome'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'mythical', 'supernatural'],
            'animation': ['animation', 'animated', 'cartoon'],
            'adventure': ['adventure', 'quest', 'exploration'],
            'war': ['war', 'military', 'battle', 'combat'],
            'western': ['western', 'cowboy', 'frontier'],
            'music': ['music', 'musical', 'concert', 'band'],
            'crime': ['crime', 'criminal', 'investigation', 'police']
        }
        
        self.content_type_keywords = {
            'movie': ['movie', 'film', 'cinema', 'flick'],
            'show': ['show', 'series', 'tv', 'television', 'season', 'episodes', 'binge', 'series to watch']
        }
        
        # Comprehensive language keywords
        self.language_keywords = {
            'Aboriginal': ['aboriginal'],
            'Afar': ['afar'],
            'Afrikaans': ['afrikaans'],
            'Akan': ['akan'],
            'Albanian': ['albanian'],
            'Amharic': ['amharic'],
            'American Sign Language': ['american sign language', 'asl'],
            'Arabic': ['arabic'],
            'Armenian': ['armenian'],
            'ASL': ['asl'],
            'Aymara': ['aymara'],
            'Azerbaijani': ['azerbaijani'],
            'Bambara': ['bambara'],
            'Basque': ['basque'],
            'Belarusian': ['belarusian'],
            'Bengali': ['bengali'],
            'Bosnian': ['bosnian'],
            'Bulgarian': ['bulgarian'],
            'Cantonese': ['cantonese'],
            'Catalan': ['catalan'],
            'Chechen': ['chechen'],
            'Chichewa': ['chichewa', 'nyanja'],
            'Cornish': ['cornish'],
            'Cree': ['cree'],
            'Croatian': ['croatian'],
            'Czech': ['czech'],
            'Danish': ['danish'],
            'Dari': ['dari'],
            'Dutch': ['dutch', 'flemish'],
            'Dzongkha': ['dzongkha'],
            'English': ['english'],
            'Esperanto': ['esperanto'],
            'Estonian': ['estonian'],
            'Faroese': ['faroese'],
            'Filipino': ['filipino', 'tagalog'],
            'Finnish': ['finnish'],
            'Flemish': ['flemish', 'dutch'],
            'French': ['french'],
            'Fulah': ['fulah'],
            'Gaelic': ['gaelic', 'irish'],
            'Galician': ['galician'],
            'Georgian': ['georgian'],
            'German': ['german'],
            'Greek': ['greek'],
            'Guarani': ['guarani'],
            'Gujarati': ['gujarati'],
            'Haitian': ['haitian', 'haitian creole'],
            'Hausa': ['hausa'],
            'Hebrew': ['hebrew'],
            'Hiligaynon': ['hiligaynon'],
            'Hindi': ['hindi', 'bollywood'],
            'Hmong': ['hmong'],
            'Hokkien': ['hokkien'],
            'Hungarian': ['hungarian'],
            'Icelandic': ['icelandic'],
            'Ilocano': ['ilocano'],
            'Indonesian': ['indonesian'],
            'Inuktitut': ['inuktitut'],
            'Inupiaq': ['inupiaq'],
            'Irish': ['irish', 'gaelic'],
            'isiZulu': ['isizulu', 'zulu'],
            'Italian': ['italian'],
            'Japanese': ['japanese', 'anime'],
            'Javanese': ['javanese'],
            'Kannada': ['kannada'],
            'Kazakh': ['kazakh'],
            'Khmer': ['khmer', 'cambodian'],
            'Kirghiz': ['kirghiz', 'kyrgyz'],
            'Korean': ['korean', 'k-drama', 'kdrama'],
            'Kurdish': ['kurdish'],
            'Latin': ['latin'],
            'Lingala': ['lingala'],
            'Lithuanian': ['lithuanian'],
            'Macedonian': ['macedonian'],
            'Malagasy': ['malagasy'],
            'Malayalam': ['malayalam'],
            'Malay': ['malay'],
            'Maltese': ['maltese'],
            'Mandarin': ['mandarin', 'chinese'],
            'Maori': ['maori'],
            'Marathi': ['marathi'],
            'Nepali': ['nepali'],
            'Norwegian': ['norwegian'],
            'Ojibwa': ['ojibwa'],
            'Oromo': ['oromo'],
            'Palawa kani': ['palawa kani'],
            'Persian': ['persian', 'farsi'],
            'Polish': ['polish'],
            'Portuguese': ['portuguese', 'brazilian'],
            'Punjabi': ['punjabi'],
            'Pushto': ['pushto', 'pashto'],
            'Quechua': ['quechua'],
            'Romani': ['romani'],
            'Romanian': ['romanian'],
            'Russian': ['russian'],
            'Saami': ['saami', 'sami'],
            'Serbian': ['serbian'],
            'Serbo-Croatian': ['serbo-croatian'],
            'Sesotho': ['sesotho', 'sotho'],
            'Setswana': ['setswana', 'tswana'],
            'Slovak': ['slovak'],
            'Somali': ['somali'],
            'Sotho': ['sotho', 'sesotho'],
            'Spanish': ['spanish', 'latino', 'hispanic'],
            'Sundanese': ['sundanese'],
            'Swahili': ['swahili'],
            'Swedish': ['swedish'],
            'Tagalog': ['tagalog', 'filipino'],
            'Tamil': ['tamil'],
            'Telugu': ['telugu'],
            'Thai': ['thai'],
            'Tswana': ['tswana', 'setswana'],
            'Turkish': ['turkish'],
            'Ukrainian': ['ukrainian'],
            'Ukrainian Sign Language': ['ukrainian sign language'],
            'Uighur': ['uighur', 'uyghur'],
            'Urdu': ['urdu'],
            'Vietnamese': ['vietnamese'],
            'Wolof': ['wolof'],
            'Xhosa': ['xhosa'],
            'Yiddish': ['yiddish'],
            'Yoruba': ['yoruba'],
            'Zulu': ['zulu', 'isizulu']
        }
        
        self.confidence_indicators = {
            'high_confidence': ['looking for', 'need', 'want', 'searching for', 'recommend', 'find me'],
            'low_confidence': ['maybe', 'perhaps', 'not sure', 'don\'t know', 'anything', 'whatever'],
            'specific_request': ['like', 'similar to', 'from year', 'starring', 'directed by', 'produced by']
        }
    
    def extract_mood_and_genre(self, query: str) -> Dict:
        """Extract mood, genre, content type, language, and confidence from user query"""
        query_lower = query.lower()
        
        # Extract moods
        detected_moods = []
        for mood, keywords in self.mood_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_moods.append(mood)
        
        # Extract genres
        detected_genres = []
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_genres.append(genre)
        
        # Extract content type preference
        preferred_content_type = None
        for content_type, keywords in self.content_type_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                preferred_content_type = content_type
                break
        
        # Extract language preference
        preferred_language = None
        for lang, keywords in self.language_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                preferred_language = lang
                break
        
        # Extract year preference
        year_preference = None
        # Look for year patterns
        year_matches = re.findall(r'\b(19|20)\d{2}\b', query_lower)
        if year_matches:
            years = [int(y) for y in year_matches]
            year_preference = (min(years), max(years))
        elif 'recent' in query_lower or 'new' in query_lower or 'latest' in query_lower:
            year_preference = (2020, 2025)
        elif 'classic' in query_lower or 'old' in query_lower:
            year_preference = (1990, 2010)
        elif '2020s' in query_lower:
            year_preference = (2020, 2025)
        elif '2010s' in query_lower:
            year_preference = (2010, 2019)
        elif '2000s' in query_lower:
            year_preference = (2000, 2009)
        elif '90s' in query_lower or 'nineties' in query_lower:
            year_preference = (1990, 1999)
        elif '80s' in query_lower or 'eighties' in query_lower:
            year_preference = (1980, 1989)
        
        # Determine confidence level
        confidence = 'medium'
        if any(indicator in query_lower for indicator in self.confidence_indicators['high_confidence']):
            confidence = 'high'
        elif any(indicator in query_lower for indicator in self.confidence_indicators['low_confidence']):
            confidence = 'low'
        
        # Check for specific requests
        has_specific_request = any(indicator in query_lower for indicator in self.confidence_indicators['specific_request'])
        
        # DEBUG: Print extraction results
        print(f"🔍 DEBUG - Query Analysis for: '{query}'")
        print(f"   Detected moods: {detected_moods}")
        print(f"   Detected genres: {detected_genres}")
        print(f"   Content type: {preferred_content_type}")
        print(f"   Language: {preferred_language}")
        print(f"   Year range: {year_preference}")
        
        return {
            'moods': detected_moods,
            'genres': detected_genres,
            'content_type': preferred_content_type,
            'preferred_language': preferred_language,
            'year_preference': year_preference,
            'confidence': confidence,
            'has_specific_request': has_specific_request,
            'query_length': len(query.split()),
            'original_query': query
        }

class EnhancedRecommendationEngine:
    """Enhanced recommendation engine with SentenceTransformers and Anthropic"""
    
    def __init__(self, netflix_data: Dict, anthropic_api_key: str = None):
        self.netflix_data = netflix_data
        self.mood_extractor = MoodExtractor()
        
        # Initialize SentenceTransformer
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SentenceTransformer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading SentenceTransformer: {e}")
            self.sentence_model = None
        
        # Initialize Anthropic client
        self.anthropic_client = None
        if anthropic_api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
                print("✅ Anthropic client initialized")
            except Exception as e:
                print(f"❌ Error initializing Anthropic: {e}")
        
        # Process data
        self.content_df = self._process_netflix_data()
        self.content_embeddings = self._create_content_embeddings()
        
        # Recommendation thresholds
        self.confidence_threshold = 0.3
        self.min_recommendations = 3
    
    def _extract_year_from_title(self, title_with_year):
        """Extract year from title_with_year field"""
        try:
            if pd.isna(title_with_year) or title_with_year is None:
                return 2020
            # Look for (YYYY) pattern at the end
            match = re.search(r'\((\d{4})\)', str(title_with_year))
            return int(match.group(1)) if match else 2020
        except:
            return 2020
    
    def _ensure_list(self, value):
        """Ensure value is a list and handle various input formats"""
        try:
            # DEBUG: Print what we're processing with more detail
            original_value = value
            print(f"🔧 _ensure_list input: '{original_value}' (type: {type(original_value)})")
            
            # Handle pandas Series/arrays first
            if hasattr(value, 'tolist'):
                value = value.tolist()
                print(f"    -> Converted pandas object to list: {value}")
            
            # Handle numpy arrays
            if hasattr(value, 'item') and hasattr(value, 'ndim'):
                if value.ndim == 0:  # scalar
                    value = value.item()
                else:
                    value = value.tolist()
                print(f"    -> Converted numpy object: {value}")
            
            # Now safe to check for None/NaN
            if value is None:
                result = []
                print(f"    -> Case: None -> {result}")
            elif isinstance(value, str) and value.lower() in ['nan', 'none', '']:
                result = []
                print(f"    -> Case: String null -> {result}")
            elif isinstance(value, list):
                # Check if it's actually a list with content
                print(f"    -> Input is list with {len(value)} items: {value}")
                result = [str(item).strip() for item in value if item is not None and str(item).strip() and str(item).lower() not in ['nan', 'none']]
                print(f"    -> Case: LIST -> {result}")
            elif isinstance(value, str):
                value = value.strip()
                if not value or value.lower() in ['nan', 'none']:
                    result = []
                    print(f"    -> Case: EMPTY STRING -> {result}")
                # Handle comma-separated values
                elif ',' in value:
                    result = [item.strip() for item in value.split(',') if item.strip() and item.strip().lower() not in ['nan', 'none']]
                    print(f"    -> Case: COMMA-SEPARATED -> {result}")
                else:
                    result = [value]
                    print(f"    -> Case: SINGLE STRING -> {result}")
            else:
                # Try to use pandas isna for other types, but safely
                try:
                    if pd.isna(value):
                        result = []
                        print(f"    -> Case: PD.ISNA -> {result}")
                    else:
                        result = [str(value).strip()] if str(value).strip() and str(value).lower() not in ['nan', 'none'] else []
                        print(f"    -> Case: OTHER ({type(value)}) -> {result}")
                except (ValueError, TypeError):
                    # Fallback if pd.isna fails
                    result = [str(value).strip()] if str(value).strip() and str(value).lower() not in ['nan', 'none'] else []
                    print(f"    -> Case: FALLBACK ({type(value)}) -> {result}")
            
            # Extra verification for important cases
            if original_value and not result:
                print(f"    ⚠️  WARNING: Had input '{original_value}' but got empty result!")
            
            return result
        except Exception as e:
            print(f"    ❌ ERROR in _ensure_list with '{value}': {e}")
            import traceback
            print(traceback.format_exc())
            # Return empty list as fallback
            return []
    
    def _process_netflix_data(self) -> pd.DataFrame:
        """Process Netflix data into DataFrame"""
        try:
            print("🔄 DEBUG - Starting data processing...")
            
            all_content = []
            
            # Process shows
            for show in self.netflix_data.get('shows', []):
                show_copy = show.copy()
                # Ensure content_type is set
                if 'content_type' not in show_copy:
                    show_copy['content_type'] = 'show'
                all_content.append(show_copy)
            
            # Process movies if available
            for movie in self.netflix_data.get('movies', []):
                movie_copy = movie.copy()
                movie_copy['content_type'] = 'movie'
                all_content.append(movie_copy)
            
            if not all_content:
                print("⚠️ No content data found")
                return pd.DataFrame()
            
            print(f"📊 DEBUG - Created dataframe with {len(all_content)} items")
            df = pd.DataFrame(all_content)
            
            # DEBUG: Check original data structure for Y tu mamá también
            print(f"📊 DEBUG - Original DataFrame columns: {list(df.columns)}")
            
            # Find Y tu mamá también specifically
            y_tu_mama_mask = df['title'].str.contains('Y tu mamá también', case=False, na=False)
            y_tu_mama_items = df[y_tu_mama_mask]
            
            if not y_tu_mama_items.empty:
                print("🎬 DEBUG - Found 'Y tu mamá también' in raw data:")
                y_tu_mama = y_tu_mama_items.iloc[0]
                print(f"    Title: {y_tu_mama.get('title')}")
                print(f"    Raw genre: {y_tu_mama.get('genre')} (type: {type(y_tu_mama.get('genre'))})")
                print(f"    Raw mood: {y_tu_mama.get('mood')} (type: {type(y_tu_mama.get('mood'))})")
                print(f"    Raw language: {y_tu_mama.get('language')} (type: {type(y_tu_mama.get('language'))})")
                print(f"    Raw director: {y_tu_mama.get('director')} (type: {type(y_tu_mama.get('director'))})")
            else:
                print("❌ DEBUG - 'Y tu mamá también' NOT found in raw data!")
                # Show a few sample titles
                sample_titles = df['title'].head(10).tolist()
                print(f"    Sample titles: {sample_titles}")
            
            # Clean basic fields
            df['title'] = df['title'].fillna('Unknown Title').astype(str)
            df['synopsis'] = df['synopsis'].fillna('').astype(str)
            df['language'] = df['language'].fillna('English').astype(str)
            
            # Extract year from title_with_year
            if 'title_with_year' in df.columns:
                df['year'] = df['title_with_year'].apply(self._extract_year_from_title)
            elif 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2020)
            else:
                df['year'] = 2020
            
            # Ensure year is integer
            df['year'] = df['year'].astype(int)
            
            # Ensure content_type is set
            if 'content_type' not in df.columns:
                df['content_type'] = 'show'  # Default fallback
            
            df['content_type'] = df['content_type'].astype(str)
            
            # DEBUG: Check data BEFORE _ensure_list processing
            print("🔄 DEBUG - Data BEFORE _ensure_list processing:")
            if not y_tu_mama_items.empty:
                y_tu_mama_row_idx = y_tu_mama_items.index[0]
                y_tu_mama_before = df.loc[y_tu_mama_row_idx]
                print(f"    Y tu mamá también genre BEFORE: {y_tu_mama_before.get('genre')} (type: {type(y_tu_mama_before.get('genre'))})")
                print(f"    Y tu mamá también mood BEFORE: {y_tu_mama_before.get('mood')} (type: {type(y_tu_mama_before.get('mood'))})")
            
            # Process list columns with robust handling
            print("🔄 DEBUG - Processing genre column...")
            df['genre'] = df['genre'].apply(self._ensure_list)
            
            print("🔄 DEBUG - Processing mood column...")
            df['mood'] = df['mood'].apply(self._ensure_list)
            
            # DEBUG: Check data AFTER _ensure_list processing
            print("🔄 DEBUG - Data AFTER _ensure_list processing:")
            if not y_tu_mama_items.empty:
                y_tu_mama_after = df.loc[y_tu_mama_row_idx]
                print(f"    Y tu mamá también genre AFTER: {y_tu_mama_after.get('genre')} (type: {type(y_tu_mama_after.get('genre'))})")
                print(f"    Y tu mamá también mood AFTER: {y_tu_mama_after.get('mood')} (type: {type(y_tu_mama_after.get('mood'))})")
            
            # Handle actors and director fields
            if 'actors' in df.columns:
                print("🔄 DEBUG - Processing actors column...")
                df['actors'] = df['actors'].apply(self._ensure_list)
            else:
                df['actors'] = [[] for _ in range(len(df))]
            
            if 'director' in df.columns:
                print("🔄 DEBUG - Processing director column...")
                # Special handling for director - it might be a string not a list
                def ensure_director_list(value):
                    if pd.isna(value) or value is None:
                        return []
                    elif isinstance(value, list):
                        return [str(item).strip() for item in value if item is not None and str(item).strip()]
                    elif isinstance(value, str):
                        return [value.strip()] if value.strip() else []
                    else:
                        return [str(value).strip()] if str(value).strip() else []
                
                df['director'] = df['director'].apply(ensure_director_list)
            else:
                df['director'] = [[] for _ in range(len(df))]
            
            # Handle other optional list fields
            for col in ['themes', 'awards', 'appears_on']:
                if col in df.columns:
                    df[col] = df[col].apply(self._ensure_list)
                else:
                    df[col] = [[] for _ in range(len(df))]
            
            # Create searchable text
            def create_searchable_text(row):
                try:
                    text_parts = [
                        str(row['title']),
                        str(row.get('synopsis', '')),
                        str(row.get('content_type', '')),
                        ' '.join(row.get('genre', [])),
                        ' '.join(row.get('mood', [])),
                        ' '.join(row.get('actors', [])),
                        ' '.join(row.get('director', [])),
                        str(row.get('language', ''))
                    ]
                    return ' '.join(text_parts).lower()
                except:
                    return str(row.get('title', 'unknown')).lower()
            
            df['searchable_text'] = df.apply(create_searchable_text, axis=1)
            
            # DEBUG: Final check of processed data
            print("✅ DEBUG - FINAL processed data check:")
            if not y_tu_mama_items.empty:
                y_tu_mama_final = df.loc[y_tu_mama_row_idx]
                print(f"    Y tu mamá también FINAL:")
                print(f"      Title: {y_tu_mama_final['title']}")
                print(f"      Genre: {y_tu_mama_final['genre']} (type: {type(y_tu_mama_final['genre'])}, len: {len(y_tu_mama_final['genre'])})")
                print(f"      Mood: {y_tu_mama_final['mood']} (type: {type(y_tu_mama_final['mood'])}, len: {len(y_tu_mama_final['mood'])})")
                print(f"      Language: {y_tu_mama_final['language']}")
                print(f"      Director: {y_tu_mama_final['director']}")
                print(f"      Year: {y_tu_mama_final['year']}")
                print(f"      Content Type: {y_tu_mama_final['content_type']}")
            
            print(f"📊 Processed {len(df)} content items")
            return df
            
        except Exception as e:
            print(f"❌ Error processing Netflix data: {e}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
    def _create_content_embeddings(self) -> np.ndarray:
        """Create sentence embeddings for all content"""
        if self.sentence_model is None or self.content_df.empty:
            return None
        
        try:
            texts = self.content_df['searchable_text'].tolist()
            embeddings = self.sentence_model.encode(texts, show_progress_bar=False)
            print(f"✅ Created embeddings for {len(embeddings)} items")
            return embeddings
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            return None
    
    def get_recommendations(self, query: str, top_k: int = 5) -> Dict:
        """Get recommendations using multiple approaches"""
        try:
            print(f"🎯 DEBUG - Getting recommendations for: '{query}'")
            
            # Extract mood and intent
            mood_analysis = self.mood_extractor.extract_mood_and_genre(query)
            
            # Get recommendations using different methods
            embedding_recs = self._get_embedding_recommendations(query, top_k * 2)
            mood_genre_recs = self._get_mood_genre_recommendations(mood_analysis, top_k * 2)
            
            print(f"🔍 DEBUG - Found {len(embedding_recs)} embedding recs, {len(mood_genre_recs)} mood/genre recs")
            
            # Combine and rank recommendations
            combined_recs = self._combine_recommendations(
                embedding_recs, mood_genre_recs, mood_analysis
            )
            
            print(f"🔍 DEBUG - Combined to {len(combined_recs)} total recommendations")
            
            # DEBUG: Print first few recommendations
            for i, rec in enumerate(combined_recs[:3]):
                print(f"   Rec {i+1}: {rec['title']}")
                print(f"      Genre: {rec.get('genre')} (type: {type(rec.get('genre'))})")
                print(f"      Mood: {rec.get('mood')} (type: {type(rec.get('mood'))})")
                print(f"      Language: {rec.get('language')}")
                print(f"      Method: {rec.get('method')}")
                print(f"      Score: {rec.get('final_score', rec.get('similarity_score'))}")
            
            # Determine if we should redirect to form
            should_redirect = self._should_redirect_to_form(combined_recs, mood_analysis)
            
            # Get Anthropic recommendations for comparison (if available)
            anthropic_recs = []
            if self.anthropic_client:
                anthropic_recs = self._get_anthropic_recommendations(query, top_k)
            
            return {
                'recommendations': combined_recs[:top_k],
                'mood_analysis': mood_analysis,
                'should_redirect_to_form': should_redirect,
                'anthropic_recommendations': anthropic_recs,
                'total_available': len(self.content_df),
                'confidence_score': self._calculate_confidence_score(combined_recs, mood_analysis),
                'content_breakdown': self._get_content_breakdown(combined_recs[:top_k])
            }
        except Exception as e:
            print(f"❌ Error in get_recommendations: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                'recommendations': [],
                'mood_analysis': {'moods': [], 'genres': [], 'original_query': query},
                'should_redirect_to_form': True,
                'anthropic_recommendations': [],
                'total_available': 0,
                'confidence_score': 0.0,
                'content_breakdown': {'movies': 0, 'shows': 0, 'total': 0}
            }
    
    def _get_embedding_recommendations(self, query: str, top_k: int) -> List[Dict]:
        """Get recommendations using sentence embeddings"""
        try:
            print(f"🔍 DEBUG - Getting embedding recommendations...")
            
            if self.sentence_model is None or self.content_embeddings is None or self.content_df.empty:
                print("   No sentence model, embeddings, or content available")
                return []
            
            # Encode query
            query_embedding = self.sentence_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.content_embeddings)[0]
            
            # Get top recommendations
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            recommendations = []
            for idx in top_indices:
                try:
                    content = self.content_df.iloc[idx]
                    
                    rec = {
                        'title': str(content['title']),
                        'year': int(content.get('year', 2020)),
                        'content_type': str(content.get('content_type', 'Unknown')),
                        'genre': list(content.get('genre', [])),
                        'mood': list(content.get('mood', [])),
                        'synopsis': str(content.get('synopsis', ''))[:200] + ('...' if len(str(content.get('synopsis', ''))) > 200 else ''),
                        'similarity_score': float(similarities[idx]),
                        'method': 'embedding',
                        'language': str(content.get('language', 'Unknown')),
                        'rating': str(content.get('rating', 'N/A')),
                        'director': list(content.get('director', [])),
                        'actors': list(content.get('actors', []))
                    }
                    
                    # DEBUG: Print recommendation details
                    print(f"   Embedding rec: {rec['title']}")
                    print(f"      Original genre from DF: {content.get('genre')} -> Rec genre: {rec['genre']}")
                    print(f"      Original mood from DF: {content.get('mood')} -> Rec mood: {rec['mood']}")
                    print(f"      Original language from DF: {content.get('language')} -> Rec language: {rec['language']}")
                    
                    recommendations.append(rec)
                except Exception as e:
                    print(f"   Error processing embedding rec {idx}: {e}")
                    continue
            
            print(f"   Generated {len(recommendations)} embedding recommendations")
            return recommendations
        except Exception as e:
            print(f"❌ Error in embedding recommendations: {e}")
            return []
    
    def _get_mood_genre_recommendations(self, mood_analysis: Dict, top_k: int) -> List[Dict]:
        """Get recommendations based on mood, genre, language, and year matching"""
        try:
            print(f"🔍 DEBUG - Getting mood/genre recommendations...")
            print(f"   Filtering criteria: {mood_analysis}")
            
            if self.content_df.empty:
                return []
            
            # Make a copy to avoid modifying original
            filtered_df = self.content_df.copy().reset_index(drop=True)
            original_count = len(filtered_df)
            
            # Filter by content type if specified
            if mood_analysis.get('content_type'):
                mask = filtered_df['content_type'] == mood_analysis['content_type']
                filtered_df = filtered_df[mask].reset_index(drop=True)
                print(f"   After content type filter ({mood_analysis['content_type']}): {len(filtered_df)} items")
            
            # Filter by language if specified
            if mood_analysis.get('preferred_language'):
                language_mask = filtered_df['language'].str.contains(
                    mood_analysis['preferred_language'], 
                    case=False, 
                    na=False, 
                    regex=False
                )
                filtered_df = filtered_df[language_mask].reset_index(drop=True)
                print(f"   After language filter ({mood_analysis['preferred_language']}): {len(filtered_df)} items")
            
            # Filter by year if specified
            if mood_analysis.get('year_preference'):
                year_min, year_max = mood_analysis['year_preference']
                year_mask = (filtered_df['year'] >= year_min) & (filtered_df['year'] <= year_max)
                filtered_df = filtered_df[year_mask].reset_index(drop=True)
                print(f"   After year filter ({year_min}-{year_max}): {len(filtered_df)} items")
            
            if len(filtered_df) == 0:
                print("   No items remain after filtering")
                return []
            
            # Calculate mood scores
            mood_scores = []
            for idx in range(len(filtered_df)):
                try:
                    content_moods = filtered_df.iloc[idx]['mood']
                    if mood_analysis['moods'] and content_moods:
                        content_moods_lower = [str(m).lower().replace('-', '_').replace(' ', '_') for m in content_moods]
                        query_moods_lower = [m.lower().replace('-', '_').replace(' ', '_') for m in mood_analysis['moods']]
                        overlap = len(set(query_moods_lower) & set(content_moods_lower))
                        score = overlap / len(mood_analysis['moods'])
                    else:
                        score = 0.5
                    mood_scores.append(score)
                except:
                    mood_scores.append(0.0)
            
            filtered_df['mood_score'] = mood_scores
            
            # Calculate genre scores
            genre_scores = []
            for idx in range(len(filtered_df)):
                try:
                    content_genres = filtered_df.iloc[idx]['genre']
                    if mood_analysis['genres'] and content_genres:
                        content_genres_lower = [str(g).lower() for g in content_genres]
                        query_genres_lower = [g.lower() for g in mood_analysis['genres']]
                        overlap = len(set(query_genres_lower) & set(content_genres_lower))
                        score = overlap / len(mood_analysis['genres'])
                    else:
                        score = 0.5
                    genre_scores.append(score)
                except:
                    genre_scores.append(0.0)
            
            filtered_df['genre_score'] = genre_scores
            
            # Calculate language scores
            language_scores = []
            for idx in range(len(filtered_df)):
                try:
                    content_language = str(filtered_df.iloc[idx]['language'])
                    if mood_analysis.get('preferred_language'):
                        if mood_analysis['preferred_language'].lower() in content_language.lower():
                            score = 1.0
                        else:
                            score = 0.0
                    else:
                        score = 0.5
                    language_scores.append(score)
                except:
                    language_scores.append(0.0)
            
            filtered_df['language_score'] = language_scores
            
            # Calculate year scores
            year_scores = []
            for idx in range(len(filtered_df)):
                try:
                    content_year = filtered_df.iloc[idx]['year']
                    if mood_analysis.get('year_preference'):
                        year_min, year_max = mood_analysis['year_preference']
                        if year_min <= content_year <= year_max:
                            score = 1.0
                        else:
                            # Penalty for being outside range
                            distance = min(abs(content_year - year_min), abs(content_year - year_max))
                            score = max(0, 1.0 - (distance / 10))  # Gradual penalty
                    else:
                        score = 0.5
                    year_scores.append(score)
                except:
                    year_scores.append(0.0)
            
            filtered_df['year_score'] = year_scores
            
            # Content type boost
            content_type_boost = 0.1 if mood_analysis.get('content_type') else 0
            
            # Calculate combined score with language and year weights
            filtered_df['combined_score'] = (
                (filtered_df['mood_score'] * 0.3) + 
                (filtered_df['genre_score'] * 0.3) +
                (filtered_df['language_score'] * 0.25) +
                (filtered_df['year_score'] * 0.15) +
                content_type_boost
            )
            
            # Sort and get top recommendations
            filtered_df = filtered_df.sort_values('combined_score', ascending=False)
            top_content = filtered_df.head(top_k)
            
            recommendations = []
            for idx in range(len(top_content)):
                try:
                    content = top_content.iloc[idx]
                    
                    rec = {
                        'title': str(content['title']),
                        'year': int(content.get('year', 2020)),
                        'content_type': str(content.get('content_type', 'Unknown')),
                        'genre': list(content.get('genre', [])),
                        'mood': list(content.get('mood', [])),
                        'synopsis': str(content.get('synopsis', ''))[:200] + ('...' if len(str(content.get('synopsis', ''))) > 200 else ''),
                        'similarity_score': float(content['combined_score']),
                        'method': 'mood_genre_language',
                        'language': str(content.get('language', 'Unknown')),
                        'rating': str(content.get('rating', 'N/A')),
                        'director': list(content.get('director', [])),
                        'actors': list(content.get('actors', []))
                    }
                    
                    # DEBUG: Print recommendation details
                    print(f"   Mood/genre rec: {rec['title']}")
                    print(f"      Original genre from DF: {content.get('genre')} -> Rec genre: {rec['genre']}")
                    print(f"      Original mood from DF: {content.get('mood')} -> Rec mood: {rec['mood']}")
                    print(f"      Combined score: {content['combined_score']:.3f}")
                    
                    recommendations.append(rec)
                except Exception as e:
                    print(f"   Error processing mood/genre rec {idx}: {e}")
                    continue
            
            print(f"   Generated {len(recommendations)} mood/genre recommendations")
            return recommendations
            
        except Exception as e:
            print(f"❌ Error in mood/genre recommendations: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def _get_anthropic_recommendations(self, query: str, top_k: int) -> List[Dict]:
        """Get recommendations using Anthropic API"""
        return []  # Simplified for now to avoid additional errors
    
    def _combine_recommendations(self, embedding_recs: List[Dict], 
                               mood_genre_recs: List[Dict], 
                               mood_analysis: Dict) -> List[Dict]:
        """Combine recommendations from different methods"""
        try:
            print(f"🔄 DEBUG - Combining recommendations...")
            
            # Weight the different approaches based on query characteristics
            has_specific_criteria = bool(
                mood_analysis['moods'] or 
                mood_analysis['genres'] or 
                mood_analysis.get('content_type') or
                mood_analysis.get('preferred_language') or
                mood_analysis.get('year_preference')
            )
            
            if has_specific_criteria:
                embedding_weight = 0.2
                mood_genre_weight = 0.8
                print(f"   Using specific criteria weights: embed={embedding_weight}, mood/genre={mood_genre_weight}")
            else:
                embedding_weight = 0.7
                mood_genre_weight = 0.3
                print(f"   Using generic weights: embed={embedding_weight}, mood/genre={mood_genre_weight}")
            
            # Combine unique recommendations
            seen_titles = set()
            combined = []
            
            # Add embedding recommendations with weights
            for rec in embedding_recs:
                if rec['title'] not in seen_titles:
                    rec['final_score'] = float(rec['similarity_score']) * embedding_weight
                    combined.append(rec)
                    seen_titles.add(rec['title'])
                    print(f"   Added embedding rec: {rec['title']} (score: {rec['final_score']:.3f})")
            
            # Add mood/genre recommendations with weights
            for rec in mood_genre_recs:
                if rec['title'] not in seen_titles:
                    rec['final_score'] = float(rec['similarity_score']) * mood_genre_weight
                    combined.append(rec)
                    seen_titles.add(rec['title'])
                    print(f"   Added mood/genre rec: {rec['title']} (score: {rec['final_score']:.3f})")
                else:
                    # If already exists, boost its score
                    for existing_rec in combined:
                        if existing_rec['title'] == rec['title']:
                            old_score = existing_rec['final_score']
                            existing_rec['final_score'] += float(rec['similarity_score']) * mood_genre_weight
                            print(f"   Boosted existing rec: {rec['title']} ({old_score:.3f} -> {existing_rec['final_score']:.3f})")
                            break
            
            # Sort by final score
            combined.sort(key=lambda x: x['final_score'], reverse=True)
            
            print(f"   Final combined list has {len(combined)} recommendations")
            return combined
        except Exception as e:
            print(f"❌ Error combining recommendations: {e}")
            return []
    
    def _should_redirect_to_form(self, recommendations: List[Dict], mood_analysis: Dict) -> bool:
        """Determine if user should be redirected to form"""
        try:
            # No recommendations found
            if len(recommendations) < self.min_recommendations:
                return True
            
            # Low confidence in recommendations
            if len(recommendations) > 0:
                scores = [float(rec.get('final_score', 0)) for rec in recommendations[:3]]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if avg_score < self.confidence_threshold:
                        return True
            
            return False
        except Exception as e:
            print(f"❌ Error in redirect logic: {e}")
            return True
    
    def _calculate_confidence_score(self, recommendations: List[Dict], mood_analysis: Dict) -> float:
        """Calculate overall confidence score for recommendations"""
        try:
            if not recommendations:
                return 0.0
            
            # Base score from top recommendations
            top_scores = [float(rec.get('final_score', 0)) for rec in recommendations[:3]]
            if not top_scores:
                return 0.0
            
            base_score = sum(top_scores) / len(top_scores)
            
            # Boost for clear intent
            intent_boost = 0.1 if (mood_analysis['moods'] or mood_analysis['genres']) else 0
            
            # Boost for content type match
            content_type_boost = 0.05 if mood_analysis.get('content_type') else 0
            
            # Boost for language specificity
            language_boost = 0.1 if mood_analysis.get('preferred_language') else 0
            
            # Boost for year specificity
            year_boost = 0.05 if mood_analysis.get('year_preference') else 0
            
            # Penalty for very generic queries
            generic_penalty = -0.1 if mood_analysis['query_length'] < 3 else 0
            
            final_score = base_score + intent_boost + content_type_boost + language_boost + year_boost + generic_penalty
            return max(0.0, min(1.0, final_score))
        except Exception as e:
            print(f"❌ Error calculating confidence: {e}")
            return 0.0
    
    def _get_content_breakdown(self, recommendations: List[Dict]) -> Dict:
        """Get breakdown of content types in recommendations"""
        try:
            if not recommendations:
                return {'movies': 0, 'shows': 0, 'total': 0}
            
            content_types = [rec.get('content_type', 'Unknown') for rec in recommendations]
            return {
                'movies': content_types.count('movie'),
                'shows': content_types.count('show'),
                'total': len(recommendations)
            }
        except Exception as e:
            print(f"❌ Error in content breakdown: {e}")
            return {'movies': 0, 'shows': 0, 'total': 0}