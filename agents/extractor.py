from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

class EntityExtractorAgent:
    def __init__(self, parsed_inputs):
        self.parsed_inputs = parsed_inputs
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def run(self, callbacks=None):
        system_prompt = "You extract business entities from discovery data."
        user_prompt = f"""
From this input, extract the following structured fields:
- Entity Name
- AUM
- Regulatory Classification
- Key Decision Makers
- Technology Platforms (OMS, EMS, Risk, Reporting)
- Operational Pain Points
- Manual Dependencies
- Trade Lifecycle Steps
- Risk Frameworks
- Reporting Requirements
- Meeting Date (from transcript header or dialogue)
Transcript:
{self.parsed_inputs['transcript']}

Questionnaire:
{self.parsed_inputs['questionnaire']}
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        result = self.llm.predict_messages(messages, callbacks=callbacks)
        return result.content