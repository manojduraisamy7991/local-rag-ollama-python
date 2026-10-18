"""Entry point.

Usage:
    python main.py index            # build the index from ./docs
    python main.py chat             # interactive chat with your docs
    python main.py ask "question"   # one-off question
"""
import sys

from rag import LLM_MODEL, answer, build_index, load_index


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"index", "chat", "ask"}:
        sys.exit(__doc__)

    command = sys.argv[1]

    if command == "index":
        print("Indexing documents...")
        count = build_index()
        print(f"Done. Saved {count} chunks to index.json")

    elif command == "ask":
        if len(sys.argv) < 3:
            sys.exit('Usage: python main.py ask "your question"')
        chunks, vectors = load_index()
        print("Answer: ", end="")
        answer(" ".join(sys.argv[2:]), chunks, vectors)

    else:  # chat
        chunks, vectors = load_index()
        print(f"Chat with your docs using {LLM_MODEL}. Type 'exit' to quit.\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit", ""}:
                break
            print("\nAnswer: ", end="")
            answer(question, chunks, vectors)
            print()


if __name__ == "__main__":
    main()