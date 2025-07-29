from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class RefinerAgent:
    def __init__(self, draft_bpr, feedback):
        self.draft_bpr = draft_bpr
        self.feedback = feedback
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def run(self, callbacks=None):
        system_prompt = "Refine BPR content based on evaluation feedback."
        user_prompt = f"""
DRAFT BPR:
{self.draft_bpr}

FEEDBACK:
{self.feedback}

Incorporate suggested improvements to enhance the document.
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        result = self.llm.predict_messages(messages, callbacks=callbacks)
        return result.content