"""Run the correctness and groundedness evaluation and upload results to LangSmith.

Run with:  python evaluate.py
"""

from insurance_assistant.evaluation import run_evaluation


def main():
    print("Running PolicyScan insurance assistant evaluation...")
    results = run_evaluation()
    print("Done. Open your LangSmith project to see the experiment.")
    print(results)


if __name__ == "__main__":
    main()

