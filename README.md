# Gene Product Description Pipeline

An LLM-powered automated pipeline for generating gene product descriptions from scientific literature, with support for multi-model comparison.

## Overview

This pipeline automates the extraction of gene-specific information from research papers and generates standardised product descriptions (PDs) using large language models (LLMs). It supports multiple workflow stages and comprehensive testing across different LLM providers.

## Features

- **Multi-stage workflow**: Summary generation → PD brainstorming → PD verification
- **Multi-model support**: Test across Claude, GPT, Gemini, DeepSeek, and more (via OpenRouter or Anthropic/OpenAI directly)
- **PubMed integration**: Automatic paper retrieval and parsing from PMC open-access papers
- **VEuPathDB integration**: Gene synonym retrieval for comprehensive searching (get aliases via API)
- **GO evidence code assignment (beta)**: Evidence code assignments are being tested
- **Comprehensive testing**: Pipeline variation testing and model comparisons enabled

## Installation

### Prerequisites

- Python 3.8+
- API keys for LLM providers (Anthropic, OpenAI, OpenRouter)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PubLLicanProject/VPDB_AI_gene_paper_summary.git
cd VPDB_AI_gene_paper_summary
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```bash
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
```

4. Prepare your data:
- Place your test set CSV in `curated_data/`

Opional - Add example PDs to `curated_data/example_PDs.txt`
NB: only needed if running PD recommendation base don set of examples. 

## Quick Start

```python
from single_pair_pipeline_workflow import run_pipeline

# Run full pipeline for a paper-gene pair
result = run_pipeline(pmid="12345678", gene_id="PF3D7_0810800")
```

## License

[PLACEHOLDER]
