"""
NLP Text Processing Pipeline for Legal Search.

Provides configurable tokenization, stemming, stop word removal,
and query expansion for improved search relevance.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─── Stop Words ──────────────────────────────────────────────────────────────

# Common English stop words (extended for legal text)
DEFAULT_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "your", "my", "me", "him", "them",
    "us", "who", "whom", "which", "what", "where", "when", "how", "not",
    "no", "nor", "so", "if", "then", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "am", "as",
    "because", "before", "between", "both", "each", "few", "more",
    "most", "other", "own", "same", "some", "such", "only", "into",
    "over", "under", "until", "up", "down", "out", "off", "once",
    "here", "there", "further", "while", "during", "through",
})

# Legal-specific terms to NEVER remove (even if they look like stop words)
LEGAL_PRESERVE = frozenset({
    "act", "order", "case", "court", "law", "rule", "right", "rights",
    "section", "article", "clause", "appeal", "judgment", "decree",
    "petition", "suit", "trial", "bail", "writ", "notice", "plea",
    "hearing", "evidence", "witness", "statute", "verdict", "sentence",
    "damages", "liability", "negligence", "breach", "contract", "tort",
    "criminal", "civil", "constitutional", "fundamental", "jurisdiction",
})


# ─── Porter Stemmer (simplified) ────────────────────────────────────────────

class PorterStemmer:
    """
    Simplified Porter stemming algorithm.
    
    Reduces words to their root form for better recall:
      'constitutional' → 'constitut'
      'judgments' → 'judgment'
      'appealed' → 'appeal'
    """

    _STEP2_SUFFIXES = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
        ("iviti", "ive"), ("biliti", "ble"),
    ]

    _STEP3_SUFFIXES = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]

    def _measure(self, stem: str) -> int:
        """Count consonant-vowel sequences (m value)."""
        cv = ""
        for ch in stem:
            if ch in "aeiou":
                cv += "v"
            else:
                cv += "c"
        # Compress runs
        compressed = re.sub(r"c+", "C", cv)
        compressed = re.sub(r"v+", "V", compressed)
        # Count VC pairs
        return compressed.count("VC")

    def _has_vowel(self, stem: str) -> bool:
        return any(ch in "aeiou" for ch in stem)

    def _ends_double_consonant(self, word: str) -> bool:
        if len(word) < 2:
            return False
        return (
            word[-1] == word[-2]
            and word[-1] not in "aeiou"
        )

    def _cvc(self, word: str) -> bool:
        """Check if word ends consonant-vowel-consonant (not w, x, y)."""
        if len(word) < 3:
            return False
        c1 = word[-3] not in "aeiou"
        v = word[-2] in "aeiou"
        c2 = word[-1] not in "aeiou" and word[-1] not in "wxy"
        return c1 and v and c2

    def stem(self, word: str) -> str:
        """Stem a single word using Porter's algorithm."""
        if len(word) <= 2:
            return word

        word = word.lower()

        # Step 1a: plurals
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif not word.endswith("ss") and word.endswith("s"):
            word = word[:-1]

        # Step 1b: -ed, -ing
        step1b_extra = False
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                word = word[:-1]
        elif word.endswith("ed"):
            stem = word[:-2]
            if self._has_vowel(stem):
                word = stem
                step1b_extra = True
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._has_vowel(stem):
                word = stem
                step1b_extra = True

        if step1b_extra:
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif self._measure(word) == 1 and self._cvc(word):
                word += "e"

        # Step 1c: y → i
        if word.endswith("y") and self._has_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2: long suffixes
        for suffix, replacement in self._STEP2_SUFFIXES:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        # Step 3
        for suffix, replacement in self._STEP3_SUFFIXES:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + replacement
                break

        # Step 4: remove long suffixes
        for suffix in [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize",
        ]:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self._measure(stem) > 1:
                    if suffix == "ion" and stem and stem[-1] in "st":
                        word = stem
                    elif suffix != "ion":
                        word = stem
                break

        # Step 5a: remove trailing 'e'
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)):
                word = stem

        # Step 5b: double consonant with m > 1
        if self._ends_double_consonant(word) and word[-1] == "l":
            if self._measure(word[:-1]) > 1:
                word = word[:-1]

        return word


# ─── Query Expansion ─────────────────────────────────────────────────────────

# Domain-specific synonym map for legal search
LEGAL_SYNONYMS: dict[str, list[str]] = {
    "murder": ["homicide", "killing", "manslaughter"],
    "homicide": ["murder", "killing"],
    "theft": ["larceny", "stealing", "robbery", "burglary"],
    "robbery": ["theft", "larceny", "dacoity"],
    "fraud": ["deceit", "misrepresentation", "deception", "cheating"],
    "negligence": ["carelessness", "recklessness", "dereliction"],
    "contract": ["agreement", "covenant", "deed"],
    "appeal": ["review", "revision", "reconsideration"],
    "bail": ["surety", "bond", "recognizance"],
    "injunction": ["restraining", "prohibition", "stay"],
    "damages": ["compensation", "reparation", "indemnity", "restitution"],
    "custody": ["detention", "confinement", "guardianship"],
    "divorce": ["dissolution", "separation", "annulment"],
    "employment": ["service", "labour", "labor", "termination"],
    "landlord": ["lessor", "owner", "proprietor"],
    "tenant": ["lessee", "occupant", "renter"],
    "constitution": ["fundamental", "constitutional"],
    "defamation": ["libel", "slander"],
    "evidence": ["proof", "testimony", "exhibit"],
    "acquittal": ["discharge", "exoneration", "absolution"],
    "conviction": ["guilty", "sentenced", "condemned"],
    "quash": ["set aside", "annul", "vacate", "overturn"],
    "writ": ["petition", "mandamus", "certiorari", "habeas corpus"],
}


def expand_query(tokens: list[str], max_expansions: int = 3) -> list[str]:
    """
    Expand query tokens with domain-specific synonyms.

    Each original token may add up to `max_expansions` synonyms.
    Expanded tokens are appended (originals come first for BM25 weighting).

    Args:
        tokens: Original query tokens.
        max_expansions: Max synonyms per token.

    Returns:
        Expanded token list (originals + synonyms).
    """
    expanded = list(tokens)
    seen = set(t.lower() for t in tokens)

    for token in tokens:
        key = token.lower()
        if key in LEGAL_SYNONYMS:
            count = 0
            for synonym in LEGAL_SYNONYMS[key]:
                syn_lower = synonym.lower()
                if syn_lower not in seen:
                    expanded.append(syn_lower)
                    seen.add(syn_lower)
                    count += 1
                    if count >= max_expansions:
                        break

    return expanded


# ─── Processing Pipeline ─────────────────────────────────────────────────────

@dataclass
class TextProcessorConfig:
    """Configuration for the text processing pipeline."""

    remove_stop_words: bool = True
    apply_stemming: bool = True
    expand_queries: bool = True
    min_token_length: int = 2
    max_token_length: int = 50
    normalize_unicode: bool = True
    preserve_legal_terms: bool = True
    custom_stop_words: frozenset = field(default_factory=frozenset)

    @property
    def stop_words(self) -> frozenset:
        """Combined stop words minus legal-preserve terms."""
        base = DEFAULT_STOP_WORDS | self.custom_stop_words
        if self.preserve_legal_terms:
            base = base - LEGAL_PRESERVE
        return base


class TextProcessor:
    """
    Configurable NLP pipeline for legal text.

    Pipeline stages:
      1. Unicode normalization & lowercasing
      2. Tokenization (word boundary extraction)
      3. Stop word removal
      4. Length filtering
      5. Stemming (optional)
      6. Query expansion (optional, query-time only)
    """

    def __init__(self, config: Optional[TextProcessorConfig] = None):
        self.config = config or TextProcessorConfig()
        self._stemmer = PorterStemmer() if self.config.apply_stemming else None

    def normalize(self, text: str) -> str:
        """Normalize text: lowercase, strip extra whitespace, clean punctuation."""
        text = text.lower()

        # Replace common legal punctuation patterns
        text = re.sub(r"[§¶†‡]", " ", text)  # legal symbols → space
        text = re.sub(r"[\u2018\u2019\u0060]", "'", text)    # normalize quotes
        text = re.sub(r'[\u201c\u201d\u201e]', '"', text)  # normalize double quotes
        text = re.sub(r"\s+", " ", text)      # collapse whitespace
        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        """Extract word tokens from normalized text."""
        text = self.normalize(text)
        tokens = re.findall(r"\b[a-z0-9]+(?:[-'][a-z0-9]+)*\b", text)

        # Length filter
        tokens = [
            t for t in tokens
            if self.config.min_token_length <= len(t) <= self.config.max_token_length
        ]

        return tokens

    def remove_stops(self, tokens: list[str]) -> list[str]:
        """Remove stop words from token list."""
        if not self.config.remove_stop_words:
            return tokens
        stops = self.config.stop_words
        return [t for t in tokens if t not in stops]

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Apply stemming to tokens."""
        if not self._stemmer:
            return tokens
        return [self._stemmer.stem(t) for t in tokens]

    def process_document(self, text: str) -> list[str]:
        """
        Full pipeline for indexing a document.

        Steps: normalize → tokenize → stop words → stem.
        """
        tokens = self.tokenize(text)
        tokens = self.remove_stops(tokens)
        tokens = self.stem_tokens(tokens)
        return tokens

    def process_query(self, query: str) -> list[str]:
        """
        Full pipeline for a search query.

        Steps: normalize → tokenize → stop words → expand → stem.
        Query expansion happens BEFORE stemming so synonyms get stemmed too.
        """
        tokens = self.tokenize(query)
        tokens = self.remove_stops(tokens)

        if self.config.expand_queries:
            tokens = expand_query(tokens)

        tokens = self.stem_tokens(tokens)
        return tokens

    def get_token_stats(self, text: str) -> dict:
        """Analyze a text and return token statistics."""
        raw_tokens = self.tokenize(text)
        filtered = self.remove_stops(raw_tokens)
        stemmed = self.stem_tokens(filtered)

        # Count unique stems
        unique_raw = set(raw_tokens)
        unique_stemmed = set(stemmed)

        return {
            "raw_count": len(raw_tokens),
            "filtered_count": len(filtered),
            "stemmed_count": len(stemmed),
            "unique_raw": len(unique_raw),
            "unique_stemmed": len(unique_stemmed),
            "stop_words_removed": len(raw_tokens) - len(filtered),
            "compression_ratio": round(
                len(unique_stemmed) / max(len(unique_raw), 1), 3
            ),
        }
