"""CLI entry point for running the insurance assistant demo."""

from insurance_assistant.pipeline import ask, build_insurance_assisstant

def main():
    """Run the insurance assistant pipeline with a set of demo questions."""
    agent = build_insurance_assisstant()
    demo_questions = [
        "Hello who are you",
        "What can you do for me?",
        "What is the waiting period?",
        "What is 20 * 20?",
        "What is co payment and how is it applied?",
        "What is an optional cover for critical illness?",
    ]

    for question in demo_questions:
        print(f"User: {question}")
        print()
        answer = ask(agent, question)
        print("Answer: ", answer)
        print()

if __name__ == "__main__":
    main()
