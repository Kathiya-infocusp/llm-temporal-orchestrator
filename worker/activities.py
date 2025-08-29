import asyncio

from temporalio import activity

from worker.llms import gemini
from worker.shared import InvoiceData


class LLMActivities:
    def __init__(self):
        self.llm = gemini()

    @activity.defn
    async def load_input(self, data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.load_input, data.fields_to_extract
            )
            return confirmation
        except Exception:
            activity.logger.exception("load input failed")
            raise

    @activity.defn
    async def construct_prompt(self, data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.construct_prompt,
                data.context_input,
            )

            return confirmation
        except Exception:
            activity.logger.exception("prompts constuction failed")
            raise

    @activity.defn
    async def call_model(self, data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.call_model
            )
            return confirmation
        except Exception:
            activity.logger.exception("call model failed")
            raise 

    @activity.defn
    async def parse_and_validate(self,data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.parse_and_validate,data.context_input, data.fields_to_extract
            )
            return confirmation
        except Exception:
            activity.logger.exception("parse_and_validate failed")
            raise


    @activity.defn
    async def persist_artifact(self, data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.parse_and_validate,data.context_input, data.fields_to_extract
            )
            return confirmation
        except Exception:
            activity.logger.exception("parse_and_validate failed")
            raise
    


