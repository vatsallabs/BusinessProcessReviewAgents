from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents.ingestor import InputIngestorAgent
from agents.extractor import EntityExtractorAgent
from agents.synthesizer import BPRSynthesizerAgent
from agents.evaluator import EvaluatorAgent
from agents.refiner import RefinerAgent

class BPRState(TypedDict):
    transcript: str
    questionnaire: str
    template: str
    entities: Optional[str]
    draft_bpr: Optional[str]
    evaluation: Optional[str]
    final_bpr: Optional[str]
    needs_refinement: Optional[bool]
    ground_truth: Optional[str]

def initialize_state(transcript_path: str, questionnaire_path: str, template_path: str, ground_truth_path: str) -> BPRState:
    with open(transcript_path) as f:
        transcript = f.read()
    with open(questionnaire_path) as f:
        questionnaire = f.read()
    with open(template_path) as f:
        template = f.read()
    with open(ground_truth_path) as f:
        ground_truth = f.read()

    return BPRState(
        transcript=transcript,
        questionnaire=questionnaire,
        template=template,
        entities=None,
        draft_bpr=None,
        evaluation=None,
        final_bpr=None,
        needs_refinement=None,
        ground_truth=ground_truth
    )

def build_bpr_graph():
    builder = StateGraph(BPRState)

    def ingest_node(state):
        return state

    def extract_node(state):
        agent = EntityExtractorAgent({
            "transcript": state["transcript"],
            "questionnaire": state["questionnaire"]
        })
        return {"entities": agent.run()}

    def synth_node(state):
        agent = BPRSynthesizerAgent(
            entity_data=state["entities"],
            template_text=state["template"]
        )
        return {"draft_bpr": agent.run()}

    def eval_node(state):
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w+", delete=False) as f:
            f.write(state["ground_truth"])
            f.flush()
            agent = EvaluatorAgent(state["draft_bpr"], f.name)
            evaluation_result = agent.run()
            needs_refine = "score: 0" in evaluation_result.lower() or "score: 1" in evaluation_result.lower()
        return {"evaluation": evaluation_result, "needs_refinement": needs_refine}

    def refine_node(state):
        agent = RefinerAgent(
            draft_bpr=state["draft_bpr"],
            feedback=state["evaluation"]
        )
        return {"final_bpr": agent.run()}

    def output_node(state):
        return {"final_bpr": state["draft_bpr"]}

    builder.add_node("ingest", ingest_node)
    builder.add_node("extract", extract_node)
    builder.add_node("synthesize", synth_node)
    builder.add_node("evaluate", eval_node)
    builder.add_node("refine", refine_node)
    builder.add_node("output", output_node)

    builder.set_entry_point("ingest")
    builder.add_edge("ingest", "extract")
    builder.add_edge("extract", "synthesize")
    builder.add_edge("synthesize", "evaluate")

    builder.add_conditional_edges(
        "evaluate", lambda state: "refine" if state["needs_refinement"] else "output"
    )
    builder.add_edge("refine", END)
    builder.add_edge("output", END)

    return builder.compile()