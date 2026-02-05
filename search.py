"""
BM25 Search Engine for Legal Cases.

Provides fast, relevant search with optional filtering.
"""

import re
from typing import Optional
from rank_bm25 import BM25Okapi


class LegalSearchEngine:
    """
    Search engine using BM25 ranking algorithm.
    
    BM25 is a bag-of-words ranking function that considers:
    - Term frequency (TF)
    - Inverse document frequency (IDF)
    - Document length normalization
    """
    
    def __init__(self):
        self.cases = []
        self.bm25 = None
        self.tokenized_corpus = []
    
    def index(self, cases: list[dict]):
        """
        Index cases for search.
        
        Args:
            cases: List of case dicts with 'text', 'headnote', 'title' fields
        """
        self.cases = cases
        
        # Build search corpus (combine searchable fields)
        corpus = []
        for case in cases:
            text = " ".join([
                case.get('title', ''),
                case.get('headnote', ''),
                case.get('text', ''),
                case.get('citation', ''),
            ])
            corpus.append(text)
        
        # Tokenize
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for search.
        
        Simple tokenization - for production consider:
        - Stemming/lemmatization
        - Legal-specific tokenization
        - Stop word removal
        """
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return tokens
    
    def search(
        self,
        query: str,
        court: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search for cases.
        
        Args:
            query: Search query
            court: Filter by court name
            year: Filter by year
            limit: Maximum results to return
        
        Returns:
            List of matching cases with relevance scores
        """
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Combine cases with scores
        scored_cases = list(zip(self.cases, scores))
        
        # Apply filters
        if court:
            scored_cases = [
                (c, s) for c, s in scored_cases
                if c.get('court', '').lower() == court.lower()
            ]
        
        if year:
            scored_cases = [
                (c, s) for c, s in scored_cases
                if c.get('year') == year or str(year) in str(c.get('date', ''))
            ]
        
        # Sort by relevance
        scored_cases.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out zero scores and limit
        results = []
        for case, score in scored_cases[:limit]:
            if score > 0:
                result = case.copy()
                result['relevance'] = round(score, 4)
                results.append(result)
        
        return results
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        courts = set()
        years = set()
        
        for case in self.cases:
            if case.get('court'):
                courts.add(case['court'])
            
            # Extract year from date or year field
            year = case.get('year')
            if not year and case.get('date'):
                try:
                    year = int(case['date'][:4])
                except (ValueError, IndexError):
                    pass
            if year:
                years.add(year)
        
        return {
            'total_cases': len(self.cases),
            'total_courts': len(courts),
            'courts': list(courts),
            'years': list(years),
        }
    
    def get_recent(self, limit: int = 10) -> list[dict]:
        """Get most recent cases by date."""
        sorted_cases = sorted(
            self.cases,
            key=lambda x: x.get('date', ''),
            reverse=True
        )
        return sorted_cases[:limit]
    
    def get_by_court(self, court: str, limit: int = 20) -> list[dict]:
        """Get cases for a specific court."""
        matches = [
            c for c in self.cases
            if c.get('court', '').lower() == court.lower()
        ]
        return matches[:limit]
