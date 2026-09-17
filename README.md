# RAGAS Evaluation Using Local Ollama

## Overview

This project is a local LLM evaluation setup built using RAGAS and Ollama.

The main goal was to evaluate RAG test cases without using an external LLM API for the evaluation. Ollama is used to run the evaluator LLM and embedding model locally, while RAGAS is used to calculate the evaluation metrics.

The project currently evaluates 56 test cases using three metrics:

* Faithfulness
* Answer Relevancy
* Context Precision

The evaluation results are saved as a CSV file along with a benchmark report.

---

## What This Project Does

Each test case contains:

* A user question
* Retrieved context
* Generated answer
* Ground-truth answer

The project loads these test cases and sends them through RAGAS for evaluation.

The evaluation flow is:

```text
Test Case Data
      |
      v
RAGAS Dataset
      |
      v
Ollama Evaluator
      |
      +-------------------+
      |                   |
      v                   v
qwen2.5:3b        nomic-embed-text
   LLM                 Embeddings
      |                   |
      +---------+---------+
                |
                v
         RAGAS Metrics
                |
       +--------+---------+
       |        |         |
       v        v         v
Faithfulness  Answer    Context
              Relevancy Precision
       |        |         |
       +--------+---------+
                |
                v
            PASS / FAIL
                |
                v
          CSV + Report
```

---

## Why Ollama?

The initial objective was to run the evaluation locally instead of depending on an external LLM API.

Ollama provides a simple way to run LLMs and embedding models locally. In this project:

* `qwen2.5:3b` is used as the evaluator LLM.
* `nomic-embed-text` is used for embeddings.

This also helped me understand how an LLM can be used as a judge for evaluating another RAG system.

---

## Technology Used

| Technology            | Purpose                             |
| --------------------- | ----------------------------------- |
| Python                | Main evaluation script              |
| RAGAS 0.4.3           | RAG evaluation                      |
| Ollama                | Local LLM runtime                   |
| qwen2.5:3b            | Evaluation LLM                      |
| nomic-embed-text      | Embedding model                     |
| Pandas                | Data processing and result handling |
| Hugging Face Datasets | Dataset preparation                 |
| Requests              | Communication with Ollama           |

---

## RAGAS Metrics

### Faithfulness

Faithfulness checks whether the generated answer is supported by the retrieved context.

Threshold used in this project:

```text
0.85
```

### Answer Relevancy

Answer Relevancy checks how relevant the generated answer is to the user's question.

Threshold used:

```text
0.80
```

### Context Precision

Context Precision checks how relevant the retrieved context is for answering the question.

Threshold used:

```text
0.80
```

---

## PASS / FAIL Logic

A test case is marked as `PASS` only when all three metrics meet their respective thresholds.

For example:

```text
Faithfulness       >= 0.85
Answer Relevancy   >= 0.80
Context Precision  >= 0.80
```

If one or more metrics are below the threshold, the test case is marked as `FAIL`.

If a metric cannot be calculated and returns an invalid value, the test case is marked as `ERROR`.

---

## Project Structure

```text
llm-ragas-testing/
│
├── data/
│   └── TestCase_Final_with_GroundTruth.csv
│
├── results/
│   ├── ragas_ollama_results.csv
│   └── ollama_benchmark_report.txt
│
├── eval_ragas_hardcode_testcase.py
|── eval_ragas_ollama.py
|── eval_ragas_ollama_openAiApi.py
├── test_ollama.py
├── requirements.txt
├── README.md
└── .gitignore
```


---

## Setup

### 1. Install Ollama

Install Ollama and make sure it is running locally.

Then download the required models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Check the installed models:

```bash
ollama list
```

---

### 2. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd llm-ragas-testing
```

---

### 3. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Check Ollama

Run:

```bash
python test_ollama.py
```

This checks the connection to Ollama and verifies that the required LLM and embedding models are working.

---

### Step 2: Run the Evaluation

Run:

```bash
python eval_ragas_ollama.py
```

The script will:

1. Load the test case data.
2. Prepare the RAGAS dataset.
3. Connect to the local Ollama models.
4. Run the selected RAGAS metrics.
5. Calculate PASS / FAIL status.
6. Generate the results CSV.
7. Generate the benchmark report.

---

## Running a Small Test

During development, I used a small number of test cases first to verify that the complete pipeline was working.

The script supports:

```python
TEST_CASE_LIMIT = 1
```

For the complete benchmark:

```python
TEST_CASE_LIMIT = None
```

This makes it easier to test changes without running the complete dataset every time.

---

## Important Implementation Detail

RAGAS expects specific embedding methods during evaluation.

The Ollama embedding model was therefore connected through a small custom adapter that provides methods such as:

```text
embed_text()
embed_query()
embed_documents()
aembed_text()
aembed_query()
aembed_documents()
```

This allowed the local `nomic-embed-text` model to work with the RAGAS evaluation pipeline.

---

## Output

The evaluation generates:

### Results CSV

```text
results/ragas_ollama_results.csv
```

This contains the individual test case results and metric scores.

### Benchmark Report

```text
results/ollama_benchmark_report.txt
```

This contains the overall benchmark summary.

---

## Key Challenges

Some of the main challenges during implementation were:

* Connecting a local Ollama LLM with RAGAS.
* Making the Ollama embedding model compatible with the RAGAS embedding interface.
* Handling structured responses expected from the evaluator LLM.
* Running a larger number of evaluation cases without stopping the complete run because of one failed evaluation.
* Keeping the evaluation results easy to analyze.

---

## What I Learned

This project helped me understand the practical side of LLM evaluation, especially:

* How RAG evaluation works.
* How RAGAS metrics are calculated.
* How an LLM can be used as an evaluator.
* How to run LLMs locally using Ollama.
* How embedding models are used during evaluation.
* How to integrate different Python libraries using adapters.
* How to build a repeatable evaluation pipeline.
* How to create benchmark results from multiple test cases.

---

## Future Improvements

Some possible improvements are:

* Add more RAGAS metrics.
* Compare different evaluator models.
* Analyze individual failed test cases in more detail.
* Generate an HTML dashboard for the evaluation results.
* Compare results between different benchmark runs.
* Integrate the evaluation into a CI/CD pipeline.
* Add automated regression checks for new RAG changes.

---


