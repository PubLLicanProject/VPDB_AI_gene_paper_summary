# VPDB_AI_gene_paper_summary

**AI Summarisation Example for Genes and Papers**  
This project demonstrates how to use AI to generate summaries of scientific literature relevant to specific genes 

---

## Requirements

- Python 3.x
- OpenAI python library
- OpenAI API key

Set your OpenAI API key as an environment variable before running the script, for example:

```bash
export OPENAI_API_KEY=your_api_key_here
```

---

## Getting Started

Clone the repository and run the script:

```bash
python main.py
```

This will generate a summary for:

- **Gene ID**: `PF3D7_1133400`  
- **PubMed ID**: `27128092`  
(Taken from user comments as the default test case)

---

## Usage

To summarize a paper for a specific gene, use the following syntax:

```bash
python main.py <gene_id> <pubmed_id>
```

### Example:

```bash
python main.py PF3D7_1133400 27128092
```

---

## 📚 Notes

- Make sure your `OPENAI_API_KEY` is set before running the script.
- For use with gene and paper data from [PlasmoDB](https://plasmodb.org) and [PubMed](https://pubmed.ncbi.nlm.nih.gov).

---

## 📄 License

MIT License. See `LICENSE` file for details.
