import os
import uuid 
import json
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from temporalio.client import Client, WorkflowExecutionStatus

from worker.workflow import InformationExtraction
from worker.shared import InvoiceData, INFORMATION_TASK_QUEUE_NAME


class TriggerRequest(BaseModel):
    context_input: str
    output: str
class TriggerResponse(BaseModel):
    workflow_id: str
class StatusResponse(BaseModel):
    workflow_id: str
    status: str
class ResultResponse(BaseModel):
    workflow_id: str
    status: str
    result: dict | None = None
    error: str | None = None




temporal_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global temporal_client
    temporal_server_url = os.getenv("TEMPORAL_GRPC_ENDPOINT", "localhost:7233")
    try:
        print(f"Connecting to Temporal server at: {temporal_server_url}")
        temporal_client = await Client.connect(temporal_server_url)
        print(f"Successfully connected to Temporal server")
    except Exception as e:
        print(f"Failed to connect to Temporal server: {e}")
        temporal_client = None 
    yield 

    if temporal_client:
        await temporal_client.close()

app = FastAPI(lifespan=lifespan, title="LLM Temporal Orchestrator")

@app.post("/workflows/trigger", response_model=TriggerResponse)
async def trigger_workflow(request: TriggerRequest):
    """
    Starts the LLM workflow with the provided question.
    """
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal client not connected")
    
    workflow_id = f"workflow-{str(uuid.uuid4())}"

    data: InvoiceData = InvoiceData(
        context_input=json.loads(request.context_input),
        fields_to_extract=[list(json.loads(_).keys()) for _ in json.loads(request.output)],
        output=[json.loads(_) for _ in json.loads(request.output)],
        workflow_id=workflow_id,
    )
    try:
        # Start the workflow
        handle = await temporal_client.start_workflow(
            InformationExtraction.run,
            data,
            id=workflow_id,
            task_queue=INFORMATION_TASK_QUEUE_NAME,
        )
        return TriggerResponse(workflow_id=handle.id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}") 
    
@app.get("/workflows/{workflow_id}/status", response_model=StatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Polls and returns the current status of the workflow.
    """
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal client not connected")

    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        description = await handle.describe()
        return StatusResponse(workflow_id=workflow_id, status=description.status.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows/{workflow_id}/result", response_model=ResultResponse)
async def get_workflow_result(workflow_id: str):
    """
    Fetches the final result of the workflow.
    If the workflow is not complete, it indicates the current status.
    """
    if not temporal_client:
        raise HTTPException(status_code=503, detail="Temporal client not connected")

    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        description = await handle.describe()

        if description.status != WorkflowExecutionStatus.COMPLETED:
            return ResultResponse(
                workflow_id=workflow_id,
                status=description.status.name,
                result=None,
                error=f"Workflow is not yet complete. Current status: {description.status.name}"
            )
        # Workflow is complete, fetch the result
        result_data = await handle.result()
        return ResultResponse(
            workflow_id=workflow_id,
            status=description.status.name,
            result=result_data,
        )

    except Exception as e:
        # This can catch application errors from within the workflow
        return ResultResponse(
            workflow_id=workflow_id,
            status="FAILED", # Assuming an exception means failure
            result=None,
            error=str(e)
        )