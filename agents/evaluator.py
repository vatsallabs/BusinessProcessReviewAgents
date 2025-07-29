from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class EvaluatorAgent:
    def __init__(self, generated_bpr, ground_truth_path):
        self.generated_bpr = generated_bpr
        self.ground_truth_path = ground_truth_path
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def run(self, callbacks=None):
        ground_truth = open(self.ground_truth_path, 'r').read()
        system_prompt = "Evaluate the similarity between generated and true BPR documents section-wise."
        user_prompt = f"""
--- GENERATED BPR ---
{self.generated_bpr}

--- GROUND TRUTH BPR ---
{ground_truth}

For each section, rate similarity 0–5 and note key differences.
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        result = self.llm.predict_messages(messages, callbacks=callbacks)
        return result.content