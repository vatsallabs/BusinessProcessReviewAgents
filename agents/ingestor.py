class InputIngestorAgent:
    def __init__(self, transcript_path, questionnaire_path, template_path):
        self.transcript_path = transcript_path
        self.questionnaire_path = questionnaire_path
        self.template_path = template_path

    def run(self, callbacks=None):
        def read_file(path):
            with open(path, 'r') as f:
                return f.read()

        return {
            "transcript": read_file(self.transcript_path),
            "questionnaire": read_file(self.questionnaire_path),
            "template_sections": read_file(self.template_path)
        }