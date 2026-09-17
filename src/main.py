import os

from dotenv import load_dotenv
from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

# IMPORTANT:
# For RAGAS 0.4.3 + evaluate(), use the metrics from ragas.metrics
# instead of ragas.metrics.collections.
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
)


# ============================================================
# 1. LOAD OPENAI API KEY
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")


# ============================================================
# 2. DEFINE RAG EVALUATION DATA
# ============================================================

question = "What is the refund policy?"

ground_truth = (
    "Customers can request a refund within 30 days of purchase."
)

answer = (
    "Customers can request a refund within 30 days of purchase."
)

retrieved_context = [
    (
        "Our refund policy allows customers to request "
        "a refund within 30 days of purchase."
    )
]


# ============================================================
# 3. CREATE RAGAS DATASET
# ============================================================

dataset = Dataset.from_list(
    [
        {
            "user_input": question,
            "retrieved_contexts": retrieved_context,
            "response": answer,
            "reference": ground_truth,
        }
    ]
)


# ============================================================
# 4. CREATE OPENAI CLIENT
# ============================================================

openai_client = OpenAI(
    api_key=API_KEY
)


# ============================================================
# 5. CREATE RAGAS EVALUATION LLM
# ============================================================

evaluator_llm = llm_factory(
    "gpt-4o-mini",
    client=openai_client,
)


# ============================================================
# 6. CREATE RAGAS EMBEDDING MODEL
# ============================================================

evaluator_embeddings = embedding_factory(
    "openai",
    model="text-embedding-3-small",
    client=openai_client,
)


# ============================================================
# 7. CREATE METRIC OBJECTS
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
# 8. RUN RAGAS EVALUATION
# ============================================================

print("\nStarting RAGAS evaluation...\n")

results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
    ],
)


# ============================================================
# 9. CONVERT RESULTS TO DATAFRAME
# ============================================================

results_df = results.to_pandas()


# ============================================================
# 10. DISPLAY INPUT DATA
# ============================================================

print("\n==============================")
print("RAGAS EVALUATION RESULTS")
print("==============================")

print("\nQuestion:")
print(question)

print("\nGround Truth:")
print(ground_truth)

print("\nAnswer:")
print(answer)

print("\nRetrieved Context:")

for i, context in enumerate(retrieved_context, 1):
    print(f"\nContext {i}:")
    print(context)


# ============================================================
# 11. DISPLAY METRICS
# ============================================================

print("\n------------------------------")
print("METRICS")
print("------------------------------")

print(
    f"Faithfulness      : "
    f"{results_df['faithfulness'].iloc[0]:.4f}"
)

print(
    f"Answer Relevancy   : "
    f"{results_df['answer_relevancy'].iloc[0]:.4f}"
)

print(
    f"Context Precision  : "
    f"{results_df['context_precision'].iloc[0]:.4f}"
)

print("\n==============================")
print("Evaluation completed successfully.")
print("==============================\n")