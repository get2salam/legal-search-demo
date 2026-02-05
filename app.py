"""
Legal Search Demo - Streamlit Application

Interactive search interface for legal case law with BM25 ranking.
"""

import json
import streamlit as st
from pathlib import Path
from search import LegalSearchEngine

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Legal Case Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .case-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 4px solid #1E88E5;
    }
    .case-title {
        font-size: 1.2em;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 5px;
    }
    .case-meta {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    .case-snippet {
        color: #333;
        line-height: 1.6;
    }
    .relevance-badge {
        background-color: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stat-number {
        font-size: 2.5em;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────────────

@st.cache_resource
def load_search_engine():
    """Load and index the case law data."""
    data_path = Path("data/cases.json")
    
    if not data_path.exists():
        # Try JSONL
        jsonl_path = Path("data/cases.jsonl")
        if jsonl_path.exists():
            cases = []
            with open(jsonl_path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        cases.append(json.loads(line))
        else:
            # Return demo data
            cases = get_sample_cases()
    else:
        with open(data_path, encoding='utf-8') as f:
            cases = json.load(f)
    
    engine = LegalSearchEngine()
    engine.index(cases)
    return engine


def get_sample_cases():
    """Generate sample cases for demo."""
    return [
        {
            "id": "demo_001",
            "title": "Constitutional Rights Case",
            "citation": "2024 SC 101",
            "court": "Supreme Court",
            "date": "2024-01-15",
            "headnote": "Landmark case on fundamental rights and due process.",
            "text": "This is a demonstration case. Add your own data to data/cases.json."
        },
        {
            "id": "demo_002",
            "title": "Contract Dispute Resolution",
            "citation": "2024 HC 205",
            "court": "High Court",
            "date": "2024-02-20",
            "headnote": "Commercial contract interpretation and breach remedies.",
            "text": "Sample case for contract law demonstration."
        },
        {
            "id": "demo_003",
            "title": "Criminal Appeal Judgment",
            "citation": "2024 CA 089",
            "court": "Court of Appeal",
            "date": "2024-03-10",
            "headnote": "Appeal against conviction on procedural grounds.",
            "text": "Criminal procedure and evidence admissibility."
        },
    ]


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    # Load search engine
    engine = load_search_engine()
    
    # ─── Sidebar ─────────────────────────────────────────────────────────────
    
    with st.sidebar:
        st.title("⚖️ Legal Search")
        st.markdown("---")
        
        # Statistics
        stats = engine.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Cases", f"{stats['total_cases']:,}")
        with col2:
            st.metric("Courts", stats['total_courts'])
        
        st.markdown("---")
        
        # Filters
        st.subheader("Filters")
        
        courts = ["All Courts"] + sorted(stats['courts'])
        selected_court = st.selectbox("Court", courts)
        
        years = ["All Years"] + sorted(stats['years'], reverse=True)
        selected_year = st.selectbox("Year", years)
        
        st.markdown("---")
        
        # Results per page
        results_per_page = st.slider("Results per page", 5, 50, 20)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit")
    
    # ─── Main Content ────────────────────────────────────────────────────────
    
    st.title("🔍 Legal Case Search")
    st.markdown("Search through case law with intelligent ranking")
    
    # Search box
    query = st.text_input(
        "Search",
        placeholder="Enter keywords, case names, or legal concepts...",
        label_visibility="collapsed"
    )
    
    # Perform search
    if query:
        # Apply filters
        court_filter = None if selected_court == "All Courts" else selected_court
        year_filter = None if selected_year == "All Years" else int(selected_year)
        
        results = engine.search(
            query,
            court=court_filter,
            year=year_filter,
            limit=results_per_page
        )
        
        st.markdown(f"### Found {len(results)} results")
        
        if results:
            for case in results:
                render_case_card(case)
        else:
            st.info("No cases found. Try different keywords or adjust filters.")
    
    else:
        # Show recent/featured cases
        st.markdown("### 📚 Browse Cases")
        st.markdown("Enter a search query above, or browse recent cases:")
        
        recent = engine.get_recent(10)
        for case in recent:
            render_case_card(case, show_relevance=False)


def render_case_card(case: dict, show_relevance: bool = True):
    """Render a case as a styled card."""
    with st.container():
        st.markdown(f"""
        <div class="case-card">
            <div class="case-title">{case.get('title', 'Untitled')}</div>
            <div class="case-meta">
                📋 {case.get('citation', 'N/A')} | 
                🏛️ {case.get('court', 'Unknown')} | 
                📅 {case.get('date', 'N/A')}
                {f' | <span class="relevance-badge">{case.get("relevance", 0):.2f}</span>' if show_relevance and 'relevance' in case else ''}
            </div>
            <div class="case-snippet">{case.get('headnote', case.get('text', '')[:300] + '...')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # View full case button
        with st.expander("View Full Case"):
            st.markdown(f"**Citation:** {case.get('citation', 'N/A')}")
            st.markdown(f"**Court:** {case.get('court', 'Unknown')}")
            st.markdown(f"**Date:** {case.get('date', 'N/A')}")
            if case.get('judges'):
                st.markdown(f"**Judges:** {', '.join(case['judges'])}")
            st.markdown("---")
            st.markdown(case.get('text', '*Full text not available*'))


if __name__ == "__main__":
    main()
