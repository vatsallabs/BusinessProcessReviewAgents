from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class BPRSynthesizerAgent:
    def __init__(self, entity_data, template_text):
        self.entity_data = entity_data
        self.template_text = template_text
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def run(self, callbacks=None):
        system_prompt = "You generate BPR documents from structured input and a template."
        user_prompt = f"""
Template:
{self.template_text}

Data:
{self.entity_data}

Fill the template to create a well-structured BPR document.
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        result = self.llm.predict_messages(messages, callbacks=callbacks)
        return result.content