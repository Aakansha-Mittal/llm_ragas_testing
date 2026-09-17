import os
import ast
import json
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbedding

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "TestCase_Final_with_GroundTruth.csv"
)

RESULTS_DIR = BASE_DIR / "results"

RESULTS_FILE = (
    RESULTS_DIR
    / "ragas_ollama_results.csv"
)

REPORT_FILE = (
    RESULTS_DIR
    / "ollama_benchmark_report.txt"
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

OLLAMA_OPENAI_URL = (
    "http://localhost:11434/v1"
)

OLLAMA_LLM_MODEL = "qwen2.5:3b"

OLLAMA_EMBEDDING_MODEL = (
    "nomic-embed-text"
)


# ============================================================
# THRESHOLDS
# ============================================================

FAITHFULNESS_THRESHOLD = 0.85

ANSWER_RELEVANCY_THRESHOLD = 0.80

CONTEXT_PRECISION_THRESHOLD = 0.80


# ============================================================
# LOCAL OLLAMA EMBEDDINGS
# ============================================================

'''class OllamaEmbeddings(BaseRagasEmbeddings):

    def __init__(
        self,
        model,
        base_url,
    ):

        self.model = model
        self.base_url = base_url.rstrip("/")


    def embed_text(
        self,
        text,
    ):

        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data["embedding"]


    def embed_query(
        self,
        text,
    ):

        return self.embed_text(text)


    def embed_documents(
        self,
        texts,
    ):

        return [
            self.embed_text(text)
            for text in texts
        ]


    async def aembed_query(
        self,
        text,
    ):

        return self.embed_query(text)


    async def aembed_documents(
        self,
        texts,
    ):

        return self.embed_documents(texts) '''

class OllamaEmbeddings(BaseRagasEmbedding):

    def __init__(
        self,
        model,
        base_url,
    ):
        super().__init__()

        self.model = model
        self.base_url = base_url.rstrip("/")


    def embed_text(
        self,
        text,
        **kwargs,
    ):

        response = requests.post(
            f"{self.base_url}/api/embeddings",

            json={
                "model": self.model,
                "prompt": text,
            },

            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data["embedding"]


    def embed_query(
        self,
        text,
        **kwargs,
    ):

        return self.embed_text(
            text,
            **kwargs,
        )


    def embed_documents(
        self,
        texts,
        **kwargs,
    ):

        return [
            self.embed_text(
                text,
                **kwargs,
            )
            for text in texts
        ]


    async def aembed_text(
        self,
        text,
        **kwargs,
    ):

        return self.embed_text(
            text,
            **kwargs,
        )


    async def aembed_query(
        self,
        text,
        **kwargs,
    ):

        return self.embed_query(
            text,
            **kwargs,
        )


    async def aembed_documents(
        self,
        texts,
        **kwargs,
    ):

        return [
            self.embed_text(
                text,
                **kwargs,
            )
            for text in texts
        ]


# ============================================================
# HELPER FUNCTION
# ============================================================

def parse_contexts(value):

    if pd.isna(value):

        return []


    if isinstance(value, list):

        return value


    value = str(value).strip()


    if not value:

        return []


    # Try Python list format

    try:

        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):

            return [
                str(item)
                for item in parsed
            ]

    except Exception:

        pass


    # Try JSON list format

    try:

        parsed = json.loads(value)

        if isinstance(parsed, list):

            return [
                str(item)
                for item in parsed
            ]

    except Exception:

        pass


    # Plain text

    return [value]


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input CSV not found:\n{INPUT_FILE}"
    )


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK OLLAMA
# ============================================================

print("\n==========================================")
print("CHECKING OLLAMA")
print("==========================================")


try:

    ollama_client = OpenAI(
        api_key="ollama",
        base_url=OLLAMA_OPENAI_URL,
    )


    response = (
        ollama_client
        .chat
        .completions
        .create(
            model=OLLAMA_LLM_MODEL,

            messages=[
                {
                    "role": "user",
                    "content":
                        "Reply with exactly: "
                        "OLLAMA SUCCESS",
                }
            ],

            temperature=0,
        )
    )


    print(
        "Ollama connection successful."
    )


    print(
        "Response:",
        response.choices[0].message.content
    )


except Exception as error:

    raise RuntimeError(
        "\nCould not connect to Ollama.\n"
        "Make sure Ollama is running "
        "and the model is installed.\n\n"
        f"Model: {OLLAMA_LLM_MODEL}\n"
        f"URL: {OLLAMA_OPENAI_URL}\n\n"
        f"Original error:\n{error}"
    )


# ============================================================
# CHECK EMBEDDING MODEL
# ============================================================

print("\n==========================================")
print("CHECKING OLLAMA EMBEDDING MODEL")
print("==========================================")


try:

    embedding_test = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",

        json={
            "model":
                OLLAMA_EMBEDDING_MODEL,

            "prompt":
                "Embedding model test",
        },

        timeout=120,
    )


    embedding_test.raise_for_status()

    embedding_data = (
        embedding_test.json()
    )


    if "embedding" not in embedding_data:

        raise ValueError(
            "Ollama did not return an embedding."
        )


    print(
        "Embedding model connection successful."
    )


    print(
        "Embedding dimensions:",
        len(
            embedding_data["embedding"]
        )
    )


except Exception as error:

    raise RuntimeError(
        "\nCould not connect to the Ollama "
        "embedding model.\n\n"
        f"Model: {OLLAMA_EMBEDDING_MODEL}\n"
        f"Original error:\n{error}"
    )


# ============================================================
# LOAD CSV
# ============================================================

print("\n==========================================")
print("LOADING TEST CASE DATA")
print("==========================================")


df = pd.read_csv(
    INPUT_FILE
)


required_columns = [
    "question",
    "contexts",
    "answer",
    "ground_truth",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


print(
    f"Test cases loaded: {len(df)}"
)


# ============================================================
# CLEAN DATA
# ============================================================

evaluation_rows = []


for index, row in df.iterrows():

    question = str(
        row["question"]
    ).strip()


    answer = str(
        row["answer"]
    ).strip()


    ground_truth = str(
        row["ground_truth"]
    ).strip()


    contexts = parse_contexts(
        row["contexts"]
    )


    if not question:

        print(
            f"Skipping row {index + 1}: "
            f"empty question"
        )

        continue


    if not answer:

        print(
            f"Skipping row {index + 1}: "
            f"empty answer"
        )

        continue


    if not ground_truth:

        print(
            f"Skipping row {index + 1}: "
            f"empty ground truth"
        )

        continue


    evaluation_rows.append(
        {
            "test_case_id":
                index + 1,

            "user_input":
                question,

            "retrieved_contexts":
                contexts,

            "response":
                answer,

            "reference":
                ground_truth,
        }
    )


print(
    f"Valid test cases for evaluation: "
    f"{len(evaluation_rows)}"
)


if not evaluation_rows:

    raise ValueError(
        "No valid test cases available "
        "for evaluation."
    )


# ============================================================
# SELECT TEST CASES FOR EVALUATION
# ============================================================

TEST_CASE_LIMIT = None

if TEST_CASE_LIMIT is None:

    evaluation_rows_to_run = (
        evaluation_rows
    )

else:

    evaluation_rows_to_run = (
        evaluation_rows[:TEST_CASE_LIMIT]
    )


# ============================================================
# CREATE RAGAS DATASET
# ============================================================

dataset = Dataset.from_list(
    evaluation_rows_to_run
)


# ============================================================
# INITIALIZE OLLAMA CLIENT
# ============================================================

print("\n==========================================")
print("INITIALIZING OLLAMA")
print("==========================================")


print(
    f"LLM Model       : "
    f"{OLLAMA_LLM_MODEL}"
)


print(
    f"Embedding Model : "
    f"{OLLAMA_EMBEDDING_MODEL}"
)


print(
    f"Base URL        : "
    f"{OLLAMA_BASE_URL}"
)


ollama_client = OpenAI(
    api_key="ollama",
    base_url=OLLAMA_OPENAI_URL,
)


# ============================================================
# RAGAS EVALUATION LLM
# ============================================================

print(
    "\nCreating Ollama evaluator LLM..."
)


evaluator_llm = llm_factory(
    OLLAMA_LLM_MODEL,
    provider="openai",
    client=ollama_client,
    temperature=0,
    max_tokens=2048,
)


# ============================================================
# RAGAS EMBEDDINGS
# ============================================================

print(
    "Creating local Ollama embedding model..."
)


evaluator_embeddings = (
    OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
)


# ============================================================
# CREATE METRICS
# ============================================================

'''faithfulness = Faithfulness(
    llm=evaluator_llm
)


answer_relevancy = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)


context_precision = ContextPrecision(
    llm=evaluator_llm,
)'''

# ============================================================
# CREATE METRICS
# ============================================================

faithfulness.llm = evaluator_llm

answer_relevancy.llm = evaluator_llm

answer_relevancy.strictness = 1

answer_relevancy.embeddings = evaluator_embeddings

context_precision.llm = evaluator_llm

 
# ============================================================
# RUN EVALUATION
# ============================================================

print("\n==========================================")
print("STARTING LOCAL RAGAS EVALUATION")
print("==========================================")


print(
    f"Total test cases: "
    f"{len(evaluation_rows_to_run)}"
)


print(
    f"Evaluation model: "
    f"{OLLAMA_LLM_MODEL}"
)


print(
    f"Embedding model: "
    f"{OLLAMA_EMBEDDING_MODEL}"
)


print(
    "\nNo OpenAI API calls will be used."
)


print(
    "\nThis may take several minutes "
    "depending on your CPU/GPU...\n"
)


results = evaluate(
    dataset=dataset,

    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
    ],

    raise_exceptions=False,

    show_progress=True,

    batch_size=1,
)


# ============================================================
# CONVERT RESULTS
# ============================================================

results_df = results.to_pandas()


# ============================================================
# MERGE ORIGINAL TEST DATA
# ============================================================

original_results = pd.DataFrame(
    evaluation_rows_to_run
)


final_df = pd.DataFrame(
    {
        "test_case_id":
            original_results[
                "test_case_id"
            ],

        "question":
            original_results[
                "user_input"
            ],

        "context":
            original_results[
                "retrieved_contexts"
            ].apply(
                lambda x:
                    "\n\n".join(x)
            ),

        "answer":
            original_results[
                "response"
            ],

        "ground_truth":
            original_results[
                "reference"
            ],
    }
)


# ============================================================
# ADD METRICS
# ============================================================

for metric_name in [

    "faithfulness",

    "answer_relevancy",

    "context_precision",

]:

    if metric_name in results_df.columns:

        final_df[
            metric_name
        ] = results_df[
            metric_name
        ].values

    else:

        final_df[
            metric_name
        ] = float("nan")


# ============================================================
# PASS / FAIL
# ============================================================

def check_pass(row):

    scores = [

        row["faithfulness"],

        row["answer_relevancy"],

        row["context_precision"],

    ]


    if any(
        pd.isna(score)
        for score in scores
    ):

        return "ERROR"


    if (

        row["faithfulness"]
        >= FAITHFULNESS_THRESHOLD

        and

        row["answer_relevancy"]
        >= ANSWER_RELEVANCY_THRESHOLD

        and

        row["context_precision"]
        >= CONTEXT_PRECISION_THRESHOLD

    ):

        return "PASS"


    return "FAIL"


final_df["status"] = final_df.apply(
    check_pass,
    axis=1
)


# ============================================================
# SAVE RESULTS
# ============================================================

final_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# CALCULATE SUMMARY
# ============================================================

total_tests = len(
    final_df
)


pass_count = (
    final_df["status"]
    == "PASS"
).sum()


fail_count = (
    final_df["status"]
    == "FAIL"
).sum()


error_count = (
    final_df["status"]
    == "ERROR"
).sum()


faithfulness_avg = (
    final_df["faithfulness"]
    .mean()
)


relevancy_avg = (
    final_df["answer_relevancy"]
    .mean()
)


precision_avg = (
    final_df["context_precision"]
    .mean()
)


# ============================================================
# PASS RATES
# ============================================================

if total_tests > 0:

    overall_pass_rate = (
        pass_count
        / total_tests
        * 100
    )

else:

    overall_pass_rate = 0


faithfulness_pass_rate = (
    final_df["faithfulness"]
    .ge(
        FAITHFULNESS_THRESHOLD
    )
    .mean()
    * 100
)


relevancy_pass_rate = (
    final_df["answer_relevancy"]
    .ge(
        ANSWER_RELEVANCY_THRESHOLD
    )
    .mean()
    * 100
)


precision_pass_rate = (
    final_df["context_precision"]
    .ge(
        CONTEXT_PRECISION_THRESHOLD
    )
    .mean()
    * 100
)


# ============================================================
# BENCHMARK REPORT
# ============================================================

report = f"""
==========================================
LOCAL OLLAMA LLM EVALUATION BENCHMARK
==========================================

Evaluation Dataset
-------------------
Total Test Cases: {total_tests}


Evaluation Model
----------------
LLM Model       : {OLLAMA_LLM_MODEL}
Embedding Model : {OLLAMA_EMBEDDING_MODEL}
Provider        : Ollama
Mode            : Local


Metrics
-------
Faithfulness Threshold     : {FAITHFULNESS_THRESHOLD}
Answer Relevancy Threshold : {ANSWER_RELEVANCY_THRESHOLD}
Context Precision Threshold: {CONTEXT_PRECISION_THRESHOLD}


Average Scores
--------------
Faithfulness      : {faithfulness_avg:.4f}
Answer Relevancy  : {relevancy_avg:.4f}
Context Precision : {precision_avg:.4f}


Test Results
------------
PASS  : {pass_count}
FAIL  : {fail_count}
ERROR : {error_count}


Overall Pass Rate
-----------------
{overall_pass_rate:.2f}%


Metric Pass Rates
-----------------

Faithfulness:
{faithfulness_pass_rate:.2f}%


Answer Relevancy:
{relevancy_pass_rate:.2f}%


Context Precision:
{precision_pass_rate:.2f}%


==========================================
END OF REPORT
==========================================
"""


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n==========================================")
print("LOCAL OLLAMA EVALUATION COMPLETED")
print("==========================================")


print(
    f"\nTotal Test Cases : "
    f"{total_tests}"
)


print(
    f"PASS             : "
    f"{pass_count}"
)


print(
    f"FAIL             : "
    f"{fail_count}"
)


print(
    f"ERROR            : "
    f"{error_count}"
)


print("\nAverage Metrics")


print(
    f"Faithfulness      : "
    f"{faithfulness_avg:.4f}"
)


print(
    f"Answer Relevancy  : "
    f"{relevancy_avg:.4f}"
)


print(
    f"Context Precision : "
    f"{precision_avg:.4f}"
)


print("\nFiles generated:")


print(
    f"Results: "
    f"{RESULTS_FILE}"
)


print(
    f"Report : "
    f"{REPORT_FILE}"
)


print("\n==========================================")