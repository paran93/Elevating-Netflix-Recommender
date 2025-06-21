import pandas as pd
import numpy as np
from typing import Dict, List
import json

class NetflixDataHandler:
    """Handle Netflix data loading and processing optimized for your JSON structure"""
    
    def __init__(self, raw_data: Dict):
        self.raw_data = raw_data
        self.processed_data = None
    
    def process_data(self) -> pd.DataFrame:
        """Process your specific Netflix JSON data structure"""
        
        # Extract shows from your JSON structure
        shows_data = self.raw_data.get('shows', [])
        
        # Convert to DataFrame
        df = pd.DataFrame(shows_data)
        
        # Clean and standardize data
        df = self._clean_data(df)
        
        self.processed_data = df
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data"""
        
        # Handle missing values
        df['title'] = df['title'].fillna('Unknown Title')
        df['synopsis'] = df['synopsis'].fillna('')
        df['review'] = df['review'].fillna('')
        df['language'] = df['language'].fillna('English')
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(2020)
        
        # Ensure list columns are properly formatted
        list_cols = ['genre', 'mood', 'themes', 'actors', 'platforms']
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._ensure_list)
        
        # Set content_type (all are shows in your data)
        df['content_type'] = 'show'
        
        # Clean up rating
        if 'mpaa_rating' in df.columns:
            df['mpaa_rating'] = df['mpaa_rating'].fillna('Not Rated')
        
        return df
    
    def _ensure_list(self, value):
        """Ensure value is a list"""
        if pd.isna(value):
            return []
        elif isinstance(value, list):
            return [item for item in value if item is not None]
        else:
            return [str(value)]
    
    def get_content_for_recommendations(self, user_query: str, n: int = 5) -> List[Dict]:
        """Get content items that could be recommended based on query"""
        
        if self.processed_data is None:
            return []
        
        # Simple content matching (in real implementation, this would be more sophisticated)
        query_lower = user_query.lower()
        
        # Score content based on simple keyword matching
        df = self.processed_data.copy()
        df['relevance_score'] = 0
        
        # Check title, synopsis, genre, mood for query keywords
        for idx, row in df.iterrows():
            score = 0
            
            # Title match
            if any(word in row['title'].lower() for word in query_lower.split()):
                score += 3
            
            # Synopsis match
            if any(word in row['synopsis'].lower() for word in query_lower.split()):
                score += 2
            
            # Genre match
            if isinstance(row['genre'], list):
                if any(word in ' '.join(row['genre']).lower() for word in query_lower.split()):
                    score += 2
            
            # Mood match
            if isinstance(row['mood'], list):
                if any(word in ' '.join(row['mood']).lower() for word in query_lower.split()):
                    score += 1
            
            df.loc[idx, 'relevance_score'] = score
        
        # Return top n relevant items
        top_content = df.nlargest(n, 'relevance_score')
        
        return top_content[['title', 'year', 'genre', 'synopsis', 'relevance_score']].to_dict('records')
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the data"""
        
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        # Get all unique genres
        all_genres = []
        for genres in df['genre'].dropna():
            if isinstance(genres, list):
                all_genres.extend(genres)
        
        # Get all unique moods
        all_moods = []
        for moods in df['mood'].dropna():
            if isinstance(moods, list):
                all_moods.extend(moods)
        
        return {
            'total_titles': len(df),
            'unique_years': df['year'].nunique(),
            'year_range': [int(df['year'].min()), int(df['year'].max())],
            'unique_languages': df['language'].nunique(),
            'languages': df['language'].value_counts().to_dict(),
            'unique_genres': len(set(all_genres)),
            'top_genres': pd.Series(all_genres).value_counts().head(10).to_dict(),
            'unique_moods': len(set(all_moods)),
            'top_moods': pd.Series(all_moods).value_counts().head(10).to_dict(),
            'avg_year': df['year'].mean()
        }