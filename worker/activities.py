import asyncio

from temporalio import activity

from worker import utils
from worker.llms import gemini
from worker.shared import InvoiceData


class LLMActivities:
    def __init__(self):
        self.llm = gemini()

    @activity.defn
    async def load_input(self, data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.load_input, data,
            )
            utils.log_structured(
                data.workflow_id, "load_input",
                attempt=activity.info().attempt, status=confirmation['status']
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
            )
            utils.log_structured(
                data.workflow_id, "construct_prompt",
                attempt=activity.info().attempt, status=confirmation['status']
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

            metadata = getattr(self.llm, 'metadata', {})
            latency_ms = getattr(self.llm, 'latency', 0)
            utils.log_structured(
                data.workflow_id, "call_model",
                attempt=activity.info().attempt, latency_ms=latency_ms,
                token_in=metadata.get('prompt_token_count', 0),
                token_out=metadata.get('candidates_token_count', 0),
                status=confirmation['status']
            )

            return confirmation
        except Exception:
            activity.logger.exception("call model failed")
            raise 

    @activity.defn
    async def parse_and_validate(self,data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.parse_and_validate,
            )
            utils.log_structured(
                data.workflow_id, "parse_and_validate",
                attempt=activity.info().attempt,
                status=confirmation['status']
            )
            return confirmation
        except Exception:
            activity.logger.exception("parse_and_validate failed")
            raise

    @activity.defn
    async def retry_model_call(self,data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.retry_model_call,
            )
            utils.log_structured(
                data.workflow_id, "retry_model_call",
                attempt=activity.info().attempt, status=confirmation['status']
            )
            return confirmation
        except Exception:
            activity.logger.exception("retry_model_call failed")
            raise

    @activity.defn
    async def persist_artifact(self, data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.persist_artifact, f"./runs/{data.workflow_id}",
            )
            utils.log_structured(
                data.workflow_id, "persist_artifact",
                path=f"./runs/{data.workflow_id}",
                attempt=activity.info().attempt, status=confirmation['status']
            )
            return confirmation
        except Exception:
            activity.logger.exception("persist_artifact failed")
            raise
    
    @activity.defn
    async def finalize(self, data: InvoiceData):
        try:
            confirmation = await asyncio.to_thread(
                self.llm.finalize,
            )
            utils.log_structured(
                data.workflow_id, "finalize",
                attempt=activity.info().attempt,
                status="success"
            )
            return confirmation
        except Exception:
            activity.logger.exception("finalize failed")
            raise
