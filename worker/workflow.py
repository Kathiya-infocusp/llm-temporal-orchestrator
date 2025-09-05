from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from worker.activities import LLMActivities
    from worker.shared import InvoiceData


@workflow.defn
class InformationExtraction:
    @workflow.run
    async def run(self, data: InvoiceData)->dict:
        retry_policy = RetryPolicy(
            maximum_attempts=1,
            maximum_interval=timedelta(seconds=2),
        )

        load_input_conformation = await workflow.execute_activity(
            LLMActivities.load_input,
            data,
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=retry_policy,
        )

        construct_prompt_confirmation = await workflow.execute_activity(
            LLMActivities.construct_prompt,
            data,
            start_to_close_timeout=timedelta(seconds=5),
            retry_policy=retry_policy,
        )

        call_model_confirmation = await workflow.execute_activity(
            LLMActivities.call_model,
            data,
            start_to_close_timeout=timedelta(seconds=360),
            retry_policy=retry_policy,
        )

        if call_model_confirmation['status'] == "failed":
            if "API call failed" in call_model_confirmation['error']:
                persist_artifact_confirmation = await workflow.execute_activity(
                    LLMActivities.persist_artifact,
                    data,
                    start_to_close_timeout=timedelta(seconds=60),
                    retry_policy=retry_policy,
                )
                return { "evalution_resul " : None, "predictions" : None}

        parse_and_validate_confirmation = await workflow.execute_activity(
            LLMActivities.parse_and_validate,
            data,
            start_to_close_timeout=timedelta(seconds=500),
            retry_policy=retry_policy,
        )

        if parse_and_validate_confirmation['status'] == "failed":
            if "Failed in validation criteria" in parse_and_validate_confirmation['error']:
                retry_model_call_confirmation = await workflow.execute_activity(
                    LLMActivities.retry_model_call,
                    data,
                    start_to_close_timeout=timedelta(seconds=1080),
                )

        finalize_confirmation = await workflow.execute_activity(
            LLMActivities.finalize,
            data,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )

        persist_artifact_confirmation = await workflow.execute_activity(
            LLMActivities.persist_artifact,
            data,
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )


        return finalize_confirmation




        

