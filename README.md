# Agentic BPR Generation Workflow (LangGraph + LangChain)

This project demonstrates an agentic workflow using LangGraph to generate a Business Process Review (BPR) document from:
- A client conversation transcript
- A discovery questionnaire
- A BPR template

The output is validated against a ground truth BPR for feedback and optional refinement.

---

## Features

- Modular agent design (Ingestor, Extractor, Synthesizer, Evaluator, Refiner)
- LangGraph-based stateful workflow with conditional refinement
- LangSmith tracing support for debugging and evaluation
- CLI interface for end-to-end execution

---

## Project Structure
agentic_bpr_poc/
├── agents/              # Individual agents (LLM-backed)

├── workflow.py          # LangGraph DAG definition

├── main.py              # Entry point CLI

├── .env.example         # LangSmith/OpenAI key management

├── requirements.txt     # Dependencies

└── README.md            # Docs

---

## Setup

1. **Clone Repo & Install Dependencies**
   uv pip install -r requirements.txt