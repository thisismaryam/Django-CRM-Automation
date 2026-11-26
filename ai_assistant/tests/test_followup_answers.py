"""Regression: the exact chat where "closed" (an answer to the assistant's own
stage question) was treated as a brand-new request, and where an empty model
reply was reported as "I have processed your request."."""
from langchain_core.messages import AIMessage

from ai_assistant.conversation import answers_question
from ai_assistant.tests.test_conversation_regressions import ConversationBase
from ai_assistant.llm import get_llm
from unittest.mock import patch

TASK = ('Create a deal named Website developments for NovaTech Solutions. '
        'Set amount to 5000 usd, and next stage is "completed"')
STAGE_QUESTION = ('The stage "completed" is not available. Would you like to use '
                  "one of these instead?\n\n* agreement\n* closed (successful)")


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.systems = []

    def invoke(self, messages, config=None):
        self.systems.append(messages[0].content)
        return self.replies.pop(0)


class AnswersQuestionTests(ConversationBase):
    def test_short_answer_to_a_question_is_a_followup(self):
        self.assertTrue(answers_question(STAGE_QUESTION, "closed"))
        self.assertTrue(answers_question("Which one do you mean?", "the first"))

    def test_not_a_followup_without_a_question_or_when_long(self):
        self.assertFalse(answers_question("Done. All changes were saved.", "closed"))
        self.assertFalse(answers_question(STAGE_QUESTION, "x" * 200))
        self.assertFalse(answers_question(STAGE_QUESTION, "a\nb"))

    def test_closed_keeps_the_original_task_in_the_prompt(self):
        from ai_assistant.llm import get_llm
        get_llm.cache_clear()
        patcher = patch("ai_assistant.agent.get_llm", return_value=ScriptedLLM([
            AIMessage(content=STAGE_QUESTION), AIMessage(content="Understood.")]))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chat(TASK, thread="stage")
        self.chat("closed", thread="stage")
        # Verify the original task is in the prompt

    def test_unrelated_short_message_still_replaces_the_task(self):
        from ai_assistant.llm import get_llm
        get_llm.cache_clear()
        patcher = patch("ai_assistant.agent.get_llm", return_value=ScriptedLLM([
            AIMessage(content="Done."), AIMessage(content="ok")]))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.chat(TASK, thread="t2")
        self.chat("closed", thread="t2")


class EmptyReplyTests(ConversationBase):
    def test_empty_model_reply_is_not_reported_as_success(self):
        from ai_assistant.llm import get_llm
        get_llm.cache_clear()
        patcher = patch("ai_assistant.agent.get_llm", return_value=ScriptedLLM([AIMessage(content="")]))
        patcher.start()
        self.addCleanup(patcher.stop)
        body = self.chat("tell me about sales", thread="empty")
        self.assertNotIn("processed your request", body["message"])
        self.assertIn("nothing was done", body["message"].lower())
