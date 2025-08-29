import asyncio
import traceback 
import uuid
import random
import json

import pandas as pd
from temporalio.client import Client, WorkflowFailureError

from worker.shared import InvoiceData, INFORMATION_TASK_QUEUE_NAME
from worker.workflow import InformationExtraction
 



async def main() -> None:
    # Create client connected to server at the given address
    client: Client = await Client.connect("localhost:7233")
    df = pd.read_excel('/home/ajaysinh/Downloads/converted_invoice_dataset.xlsx')
    df_sample = df[["Input","Final_Output"]].iloc[random.randint(1, 10)]

    workflow_id = f"workflow-{str(uuid.uuid4())}"

    data: InvoiceData = InvoiceData(
        context_input=df_sample['Input'],
        fields_to_extract=list(json.loads(df_sample['Final_Output']).keys()),
        output=json.loads(df_sample['Final_Output']),
        workflow_id=workflow_id,
    )
    # print("data : ",data)
    try:
        result = await client.execute_workflow(
            InformationExtraction.run,
            data,
            id=workflow_id,
            task_queue=INFORMATION_TASK_QUEUE_NAME,
        )

        print(f"Result: {result}")

    except WorkflowFailureError:
        print("Got expected exception: ", traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
# @@@SNIPEND
