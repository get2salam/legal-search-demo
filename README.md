# Legal Search Demo 🔍⚖️

Interactive Streamlit demo for legal case law search. Features BM25 ranking, filters, and a clean UI for exploring case databases.

## Features

- 🔍 **Full-text search** with BM25 ranking
- 🏛️ **Filter by court** and year
- 📊 **Statistics dashboard**
- 📄 **Case viewer** with syntax highlighting
- 🎨 **Clean, responsive UI**
- 🚀 **Fast** — runs entirely in browser

## Demo

![Legal Search Demo](demo.gif)

## Quick Start

```bash
# Clone and install
git clone https://github.com/get2salam/legal-search-demo.git
cd legal-search-demo
pip install -r requirements.txt

# Run
streamlit run app.py
```

Visit `http://localhost:8501` to use the demo.

## Data Format

Place your case law data in `data/cases.json`:

```json
[
  {
    "id": "case_001",
    "title": "Smith v. State",
    "citation": "2024 SC 445",
    "court": "Supreme Court",
    "date": "2024-03-15",
    "headnote": "Brief summary of the case...",
    "text": "Full judgment text..."
  }
]
```

Or use JSONL format (`data/cases.jsonl`):

```jsonl
{"id": "case_001", "title": "Smith v. State", ...}
{"id": "case_002", "title": "Jones v. City", ...}
```

## How It Works

### BM25 Search

The app uses [rank-bm25](https://github.com/dorianbrown/rank_bm25) for relevance ranking:

1. Tokenizes query and documents
2. Computes BM25 scores
3. Returns top-k results sorted by relevance

### Architecture

```
legal-search-demo/
├── app.py           # Main Streamlit application
├── search.py        # BM25 search engine
├── data/
│   └── cases.json   # Your case law data
└── requirements.txt
```

## Configuration

Edit `config.py` or set environment variables:

```python
# config.py
DATA_PATH = "data/cases.json"  # or .jsonl
RESULTS_PER_PAGE = 20
SNIPPET_LENGTH = 300
```

## Customization

### Adding Courts

The app auto-detects courts from your data. No configuration needed.

### Styling

Modify the Streamlit theme in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#212121"
```

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Deploy!

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
docker build -t legal-search-demo .
docker run -p 8501:8501 legal-search-demo
```

## Performance

- Loads ~10,000 cases in <2 seconds
- Search latency <100ms for typical queries
- Memory: ~50MB for 10,000 cases

For larger datasets (100k+ cases), consider:
- Elasticsearch backend
- Pre-computed embeddings
- Server-side pagination

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE)

---

Built with ⚖️ by [Abdul Salam](https://github.com/get2salam)
