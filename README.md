# llama-dashboard

[![PyPI version](https://badge.fury.io/py/llama-dashboard.svg)](https://badge.fury.io/py/llama-dashboard)
[![Python Version](https://img.shields.io/pypi/pyversions/llama-dashboard.svg)](https://pypi.org/project/llama-dashboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yourusername/llama-dashboard-pkg/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/llama-dashboard-pkg/actions/workflows/ci.yml)

## Installation

```bash
pip install -e .
```

## Usage

```python
from llama_dashboard import LlamaDashboardClient

# Initialize the client
client = LlamaDashboardClient(api_key="your-api-key")
result = client.query("your query")
print(result)
```

## Features

- Fast and efficient
- Easy to use API
- Comprehensive documentation
- Asynchronous support
- Built-in caching

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/nikjois/llama-dashboard.git
cd llama-dashboard

# Install development dependencies
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

## License

MIT

## Author

Nik Jois (nikjois@llamasearch.ai)
