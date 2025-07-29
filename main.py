# agentic_bpr_poc/main.py

import argparse
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain.callbacks import LangChainTracer
from workflow import build_bpr_graph, initialize_state
from tqdm import tqdm
from IPython.display import display, Image

load_dotenv()

# LangSmith setup
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
USE_LANGSMITH = os.getenv("USE_LANGSMITH", "false").lower() == "true"

tracer = LangChainTracer(project_name=LANGCHAIN_PROJECT) if USE_LANGSMITH else None


def main(args):
    stages = [
        "Building LangGraph",
        "Initializing State",
        "Invoking Workflow",
        "Saving Output",
        "Displaying Evaluation"
    ]

    with tqdm(total=len(stages), desc="BPR Generation Workflow", unit="stage") as progress:
        # Build LangGraph
        graph = build_bpr_graph()
        print("\nGraph built successfully!")
        png_graph = graph.get_graph().draw_mermaid_png()
        with open("bpr_process_graph.png", "wb") as f:
            f.write(png_graph)
        progress.update(1)

        # Initialize state
        state = initialize_state(
            transcript_path=args.transcript,
            questionnaire_path=args.questionnaire,
            template_path=args.template,
            ground_truth_path=args.ground_truth
        )
        progress.update(1)

        # Run the graph
        final_state = graph.invoke(state, config={"callbacks": [tracer] if tracer else []})
        progress.update(1)

        # Save output
        with open(args.output, "w") as f:
            f.write(final_state["final_bpr"])
        progress.update(1)

        # Show evaluation summary
        print("\nEvaluation Summary:\n")
        print(final_state.get("evaluation", "[No evaluation present]"))

        if "evaluation_scores" in final_state:
            print("\nSection-wise Scores:")
            for section, score in final_state["evaluation_scores"].items():
                print(f" - {section}: {score}/5")

        progress.update(1)
        print(f"\nFinal BPR document saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BPR using LangGraph agentic workflow")
    parser.add_argument("--transcript", required=True, help="Path to transcript file")
    parser.add_argument("--questionnaire", required=True, help="Path to discovery questionnaire")
    parser.add_argument("--template", required=True, help="Path to BPR template")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth BPR")
    parser.add_argument("--output", default="output_bpr.txt", help="Output BPR filename")
    args = parser.parse_args()
    main(args)