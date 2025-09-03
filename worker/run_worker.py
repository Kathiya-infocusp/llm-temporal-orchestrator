
import asyncio
import os 
from temporalio.client import Client
from temporalio.worker import Worker

from worker.activities import LLMActivities
from worker.shared import INFORMATION_TASK_QUEUE_NAME
from worker.workflow import InformationExtraction


async def main() -> None:
    temporal_server_url = os.getenv("TEMPORAL_GRPC_ENDPOINT", "localhost:7233")
    print(f"Connecting to Temporal server at: {temporal_server_url}")
    client: Client = await Client.connect(temporal_server_url, namespace="default")
    # Run the worker
    activities = LLMActivities()
    worker: Worker = Worker(
        client,
        task_queue=INFORMATION_TASK_QUEUE_NAME,
        workflows=[InformationExtraction],
        activities=[activities.load_input, activities.construct_prompt, activities.call_model, activities.parse_and_validate,activities.retry_model_call , activities.persist_artifact, activities.finalize],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
