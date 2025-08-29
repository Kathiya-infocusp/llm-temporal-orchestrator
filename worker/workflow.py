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
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=retry_policy,
        )

        parse_and_validate_confirmation = await workflow.execute_activity(
            LLMActivities.parse_and_validate,
            data,
            start_to_close_timeout=timedelta(seconds=500),
            retry_policy=retry_policy,
        )


        return parse_and_validate_confirmation




        

