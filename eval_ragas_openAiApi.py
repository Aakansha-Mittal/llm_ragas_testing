import os
import ast
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "TestCase_Final_with_GroundTruth.csv"

RESULTS_DIR = BASE_DIR / "results"

RESULTS_FILE = RESULTS_DIR / "ragas_results.csv"

REPORT_FILE = RESULTS_DIR / "benchmark_report.txt"

FAITHFULNESS_THRESHOLD = 0.85

ANSWER_RELEVANCY_THRESHOLD = 0.80

CONTEXT_PRECISION_THRESHOLD = 0.80


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. "
        "Please add it to the .env file."
    )


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
            return [str(item) for item in parsed]

    except Exception:
        pass

    # Try JSON list format
    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return [str(item) for item in parsed]

    except Exception:
        pass

    # If it is plain text, treat it as one context
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
# LOAD CSV
# ============================================================

print("\n==========================================")
print("LOADING TEST CASE DATA")
print("==========================================")

df = pd.read_csv(INPUT_FILE)

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
        f"Missing required columns: {missing_columns}"
    )


print(f"Test cases loaded: {len(df)}")


# ============================================================
# CLEAN DATA
# ============================================================

evaluation_rows = []

for index, row in df.iterrows():

    question = str(row["question"]).strip()

    answer = str(row["answer"]).strip()

    ground_truth = str(row["ground_truth"]).strip()

    contexts = parse_contexts(row["contexts"])

    if not question:
        print(f"Skipping row {index + 1}: empty question")
        continue

    if not answer:
        print(f"Skipping row {index + 1}: empty answer")
        continue

    if not ground_truth:
        print(
            f"Skipping row {index + 1}: empty ground truth"
        )
        continue

    evaluation_rows.append(
        {
            "test_case_id": index + 1,
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": ground_truth,
        }
    )


print(
    f"Valid test cases for evaluation: "
    f"{len(evaluation_rows)}"
)


# ============================================================
# CREATE RAGAS DATASET
# ============================================================

dataset = Dataset.from_list(
    evaluation_rows
)


# ============================================================
# OPENAI CLIENT
# ============================================================

print("\n==========================================")
print("INITIALIZING OPENAI")
print("==========================================")

openai_client = OpenAI(
    api_key=API_KEY
)


# ============================================================
# RAGAS EVALUATION LLM
# ============================================================

print("Creating evaluator LLM...")

evaluator_llm = llm_factory(
    "gpt-4o-mini",
    client=openai_client,
)


# ============================================================
# RAGAS EMBEDDINGS
# ============================================================

print("Creating embedding model...")

evaluator_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=openai_client,
)


# ============================================================
# CREATE METRICS
# ============================================================

faithfulness = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

context_precision = ContextPrecision(
    llm=evaluator_llm,
)


# ============================================================
# RUN EVALUATION
# ============================================================

print("\n==========================================")
print("STARTING RAGAS EVALUATION")
print("==========================================")

print(
    f"Total test cases: {len(evaluation_rows)}"
)

print("\nThis may take several minutes...\n")


results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
    ],
    raise_exceptions=False,
    show_progress=True,
    batch_size=5,
)


# ============================================================
# CONVERT RESULTS
# ============================================================

results_df = results.to_pandas()


# ============================================================
# MERGE ORIGINAL TEST DATA
# ============================================================

original_results = pd.DataFrame(
    evaluation_rows
)

final_df = pd.DataFrame(
    {
        "test_case_id": original_results[
            "test_case_id"
        ],
        "question": original_results[
            "user_input"
        ],
        "context": original_results[
            "retrieved_contexts"
        ].apply(lambda x: "\n\n".join(x)),
        "answer": original_results[
            "response"
        ],
        "ground_truth": original_results[
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

        final_df[metric_name] = results_df[
            metric_name
        ].values

    else:

        final_df[metric_name] = float("nan")


# ============================================================
# PASS / FAIL
# ============================================================

def check_pass(row):

    scores = [
        row["faithfulness"],
        row["answer_relevancy"],
        row["context_precision"],
    ]

    if any(pd.isna(score) for score in scores):
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

total_tests = len(final_df)

pass_count = (
    final_df["status"] == "PASS"
).sum()

fail_count = (
    final_df["status"] == "FAIL"
).sum()

error_count = (
    final_df["status"] == "ERROR"
).sum()


faithfulness_avg = (
    final_df["faithfulness"].mean()
)

relevancy_avg = (
    final_df["answer_relevancy"].mean()
)

precision_avg = (
    final_df["context_precision"].mean()
)


# ============================================================
# BENCHMARK REPORT
# ============================================================

report = f"""
==========================================
LLM EVALUATION BENCHMARK REPORT
==========================================

Evaluation Dataset
-------------------
Total Test Cases: {total_tests}

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
{(pass_count / total_tests * 100):.2f}%


Metric Pass Rates
-----------------

Faithfulness:
{(
    final_df["faithfulness"]
    .ge(FAITHFULNESS_THRESHOLD)
    .mean()
    * 100
):.2f}%

Answer Relevancy:
{(
    final_df["answer_relevancy"]
    .ge(ANSWER_RELEVANCY_THRESHOLD)
    .mean()
    * 100
):.2f}%

Context Precision:
{(
    final_df["context_precision"]
    .ge(CONTEXT_PRECISION_THRESHOLD)
    .mean()
    * 100
):.2f}%


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
print("EVALUATION COMPLETED")
print("==========================================")

print(f"\nTotal Test Cases : {total_tests}")

print(f"PASS             : {pass_count}")

print(f"FAIL             : {fail_count}")

print(f"ERROR            : {error_count}")


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
    f"Results: {RESULTS_FILE}"
)

print(
    f"Report : {REPORT_FILE}"
)

print("\n==========================================")