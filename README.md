# Gene Product Description Pipeline

An LLM-powered automated pipeline for generating gene product descriptions from scientific literature, with support for multi-model comparison.

## Overview

This pipeline automates the extraction of gene-specific information from research papers and generates standardised product descriptions (PDs) using large language models (LLMs). It supports multiple workflow stages and comprehensive testing across different LLM providers.

## Features

- **Multi-stage workflow**: Summary generation → PD brainstorming → PD verification + selection (with optional summary verification and swap-out selection criteria)
- **Multi-model support**: Test across Claude, GPT, Gemini, DeepSeek, and more (via OpenRouter or Anthropic/OpenAI directly)
- **PubMed integration**: Automatic paper retrieval and parsing from PMC open-access papers
- **VEuPathDB integration**: Gene synonym retrieval for comprehensive searching (get aliases via API)
- **GO evidence code assignment (beta)**: Evidence code assignments are being tested
- **Comprehensive testing**: Pipeline variation testing and model comparisons enabled

## Installation

### Prerequisites

- Python 3.x+
- API keys for LLM providers you intend to use (Supported: Anthropic, OpenAI, OpenRouter)

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

4. (Optional) - If you intend to run tests, prepare your data:
- Place your test set CSV with ground truth gene-paper pairs in `curated_data/`

5. (Opional) - Add example PDs to `curated_data/example_PDs.txt`
NB: only needed if running PD recommendation based on set of examples. 


## Quick Start

### Python API

```python
from main import run_pipeline

# Run full pipeline for a paper-gene pair
run_pipeline(pmid="12345678", gene_id="PF3D7_0810800", host_db="plasmodb")

# Summary generation only (skip PD stages)
run_pipeline(pmid="12345678", gene_id="PF3D7_0810800", host_db="plasmodb", generate_pd=False)
```

### Command Line

```bash
# Single pair – full pipeline
python main.py single --pmid 12345678 --gene_id PF3D7_0810800 --host_db plasmodb

# Single pair – summary only
python main.py single --pmid 12345678 --gene_id PF3D7_0810800 --host_db plasmodb --no-generate-pd

```
## Post-processing
To extract results and score PDs, the scripts in post_processing/ can be used

## Online tool hosted on HPC
Back-end is slightly outdated, but code is in PubGene_back_end for reference 

## License
MIT License

Copyright (c) 2025 tonyatliv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
