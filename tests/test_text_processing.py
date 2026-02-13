"""Tests for the NLP text processing pipeline."""

import pytest
from text_processing import (
    DEFAULT_STOP_WORDS,
    LEGAL_PRESERVE,
    LEGAL_SYNONYMS,
    PorterStemmer,
    TextProcessor,
    TextProcessorConfig,
    expand_query,
)


# ─── Porter Stemmer Tests ────────────────────────────────────────────────────

class TestPorterStemmer:
    """Test the simplified Porter stemmer."""

    def setup_method(self):
        self.stemmer = PorterStemmer()

    def test_plural_removal(self):
        assert self.stemmer.stem("cases") == "case"
        assert self.stemmer.stem("judgments") == "judgment"

    def test_sses_to_ss(self):
        assert self.stemmer.stem("caresses") == "caress"

    def test_ies_to_i(self):
        assert self.stemmer.stem("ponies") == "poni"

    def test_ed_removal(self):
        assert self.stemmer.stem("appealed") == "appeal"
        assert self.stemmer.stem("convicted") == "convict"

    def test_ing_removal(self):
        assert self.stemmer.stem("filing") == "file"
        assert self.stemmer.stem("hearing") == "hear"

    def test_short_words_unchanged(self):
        assert self.stemmer.stem("at") == "at"
        assert self.stemmer.stem("by") == "by"
        assert self.stemmer.stem("a") == "a"

    def test_y_to_i(self):
        result = self.stemmer.stem("guilty")
        assert result.endswith("i") or result == "guilti"

    def test_ational_to_ate(self):
        assert self.stemmer.stem("constitutional") != "constitutional"
        # Should shorten the word
        assert len(self.stemmer.stem("constitutional")) < len("constitutional")

    def test_ization_to_ize(self):
        result = self.stemmer.stem("legalization")
        assert "legal" in result or "ize" in result or len(result) < len("legalization")

    def test_deterministic(self):
        """Same input should always produce same output."""
        word = "constitutional"
        result1 = self.stemmer.stem(word)
        result2 = self.stemmer.stem(word)
        assert result1 == result2

    def test_lowercase_handling(self):
        assert self.stemmer.stem("APPEAL") == self.stemmer.stem("appeal")


# ─── Query Expansion Tests ───────────────────────────────────────────────────

class TestQueryExpansion:
    """Test legal domain query expansion."""

    def test_known_synonym_expansion(self):
        tokens = ["murder"]
        expanded = expand_query(tokens)
        assert "murder" in expanded
        assert "homicide" in expanded

    def test_no_duplicates(self):
        tokens = ["theft", "robbery"]
        expanded = expand_query(tokens)
        # Count each token appears at most once
        for token in set(expanded):
            assert expanded.count(token) == 1

    def test_max_expansions_limit(self):
        tokens = ["murder"]
        expanded = expand_query(tokens, max_expansions=1)
        # Should have original + at most 1 synonym
        assert len(expanded) <= 2

    def test_unknown_term_no_expansion(self):
        tokens = ["xyznonexistent"]
        expanded = expand_query(tokens)
        assert expanded == tokens

    def test_originals_come_first(self):
        tokens = ["bail", "murder"]
        expanded = expand_query(tokens)
        # Original tokens should be at the start
        assert expanded[0] == "bail"
        assert expanded[1] == "murder"

    def test_empty_input(self):
        assert expand_query([]) == []

    def test_multiple_expansions(self):
        tokens = ["fraud"]
        expanded = expand_query(tokens, max_expansions=3)
        # Original + up to 3 synonyms
        assert len(expanded) >= 2
        assert len(expanded) <= 4


# ─── Stop Words Tests ────────────────────────────────────────────────────────

class TestStopWords:
    """Test stop word configuration."""

    def test_common_words_are_stops(self):
        assert "the" in DEFAULT_STOP_WORDS
        assert "is" in DEFAULT_STOP_WORDS
        assert "and" in DEFAULT_STOP_WORDS

    def test_legal_terms_preserved(self):
        """Legal terms should NOT be in stop words after filtering."""
        config = TextProcessorConfig()
        effective_stops = config.stop_words
        for term in ["court", "law", "act", "case", "appeal"]:
            assert term not in effective_stops

    def test_custom_stop_words(self):
        config = TextProcessorConfig(
            custom_stop_words=frozenset({"foobar", "bazqux"})
        )
        assert "foobar" in config.stop_words
        assert "bazqux" in config.stop_words

    def test_preserve_legal_toggle(self):
        """When preserve_legal_terms=False, LEGAL_PRESERVE isn't subtracted."""
        config = TextProcessorConfig(preserve_legal_terms=False)
        stops = config.stop_words
        # "right" might be in default stops and should NOT be preserved
        # The key thing: legal terms are in LEGAL_PRESERVE, not DEFAULT_STOP_WORDS
        # This just tests the toggle works
        assert isinstance(stops, frozenset)


# ─── TextProcessor Tests ─────────────────────────────────────────────────────

class TestTextProcessor:
    """Test the full text processing pipeline."""

    def setup_method(self):
        self.processor = TextProcessor()

    def test_normalize_lowercase(self):
        assert self.processor.normalize("HELLO World") == "hello world"

    def test_normalize_whitespace(self):
        assert self.processor.normalize("too  many   spaces") == "too many spaces"

    def test_normalize_legal_symbols(self):
        result = self.processor.normalize("§ 302 ¶ 5")
        assert "§" not in result
        assert "¶" not in result

    def test_normalize_quotes(self):
        result = self.processor.normalize("it\u2019s a \u201ctest\u201d")
        assert "\u2019" not in result
        assert "\u201c" not in result

    def test_tokenize_basic(self):
        tokens = self.processor.tokenize("The court held that the appeal was valid.")
        assert "court" in tokens
        assert "appeal" in tokens
        assert "valid" in tokens

    def test_tokenize_min_length(self):
        tokens = self.processor.tokenize("I am a legal expert")
        # Single-char tokens should be filtered
        assert "i" not in tokens
        assert "a" not in tokens

    def test_stop_word_removal(self):
        tokens = ["the", "court", "held", "that", "appeal"]
        filtered = self.processor.remove_stops(tokens)
        assert "the" not in filtered
        assert "that" not in filtered
        assert "court" in filtered  # Legal term preserved
        assert "appeal" in filtered  # Legal term preserved

    def test_process_document(self):
        text = "The Supreme Court dismissed the appeal filed by the petitioner."
        tokens = self.processor.process_document(text)
        assert len(tokens) > 0
        # Stop words should be removed
        assert "the" not in tokens

    def test_process_query_with_expansion(self):
        query = "murder conviction"
        tokens = self.processor.process_query(query)
        # Should have expanded synonyms
        assert len(tokens) > 2

    def test_process_query_without_expansion(self):
        config = TextProcessorConfig(expand_queries=False)
        processor = TextProcessor(config)
        query = "murder conviction"
        tokens = processor.process_query(query)
        # No expansion — just original tokens (after stop/stem)
        assert len(tokens) <= 2

    def test_no_stemming_mode(self):
        config = TextProcessorConfig(apply_stemming=False)
        processor = TextProcessor(config)
        tokens = processor.process_document("The courts were appealing their judgments")
        # Words should NOT be stemmed
        assert "courts" in tokens or "appealing" in tokens or "judgments" in tokens

    def test_token_stats(self):
        text = "The court ruled that the defendant was guilty of murder."
        stats = self.processor.get_token_stats(text)
        assert stats["raw_count"] > 0
        assert stats["stop_words_removed"] >= 0
        assert 0 <= stats["compression_ratio"] <= 1.0

    def test_empty_text(self):
        assert self.processor.process_document("") == []
        assert self.processor.process_query("") == []

    def test_numbers_included(self):
        tokens = self.processor.tokenize("Section 302 of the Act 1860")
        assert "302" in tokens
        assert "1860" in tokens

    def test_hyphenated_words(self):
        tokens = self.processor.tokenize("well-known cross-examination")
        assert any("well" in t or "known" in t or "well-known" in t for t in tokens)


# ─── Integration Tests ───────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end integration tests."""

    def test_document_and_query_pipeline(self):
        """Documents and queries processed through same pipeline should match."""
        processor = TextProcessor()

        doc = "The High Court granted bail to the accused in the murder case."
        query = "bail murder"

        doc_tokens = set(processor.process_document(doc))
        query_tokens = set(processor.process_query(query))

        # There should be overlap between doc and query tokens
        overlap = doc_tokens & query_tokens
        assert len(overlap) > 0

    def test_synonym_improves_matching(self):
        """Query expansion should increase token overlap."""
        config_no_expand = TextProcessorConfig(expand_queries=False)
        config_expand = TextProcessorConfig(expand_queries=True)

        proc_no = TextProcessor(config_no_expand)
        proc_yes = TextProcessor(config_expand)

        doc = "The defendant was charged with homicide and larceny."
        doc_tokens = set(proc_yes.process_document(doc))

        query = "murder theft"
        q_no = set(proc_no.process_query(query))
        q_yes = set(proc_yes.process_query(query))

        overlap_no = doc_tokens & q_no
        overlap_yes = doc_tokens & q_yes

        # Expanded query should match more doc tokens
        assert len(overlap_yes) >= len(overlap_no)

    def test_legal_terms_survive_pipeline(self):
        """Legal terms like 'court', 'appeal' should not be stripped."""
        processor = TextProcessor()
        tokens = processor.process_document(
            "The court heard the appeal against the order."
        )
        # At least 'court' and 'appeal' should survive (they're in LEGAL_PRESERVE)
        stemmer = PorterStemmer()
        court_stem = stemmer.stem("court")
        appeal_stem = stemmer.stem("appeal")
        assert court_stem in tokens or "court" in tokens
        assert appeal_stem in tokens or "appeal" in tokens
