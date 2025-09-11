# LLM Service Orchestrated with Temporal (File-Based Storage)

A Python service that answers questions by orchestrating an LLM workflow with Temporal.

---

## Quick Start

### Set up environment:
```bash
# Add your Gemini API key to .env
cp .env.example .env

# set up dependencies using uv
uv sync

```
## Reproducible Local Run

### Option 1: local run (Recommended)
```bash
bash ./run_local.sh
```

### Option 2: Local Temporal Server
```bash
# Start Temporal server
temporal server start-dev

# In separate terminals:
python -m worker.run_worker  # Start worker
python -m api.main     # Start API
```


### Access the services:
 * **Temporal UI**: http://localhost:8233
 * **API**: http://localhost:8000
 * **API Docs**: http://localhost:8000/docs


### API Usage Examples
 Start a workflow
```bash
curl -X POST "http://localhost:8000/workflows/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "context_input": "",
    "output": {}
  }'

```
### Check workflow status
```bash
curl "http://localhost:8000/workflows/{workflow_id}/status"
```
### Get workflow result
```bash
curl "http://localhost:8000/workflows/{workflow_id}/result"
```


## Observability

### Structured Logging
JSON logs with activity metrics:
```json
{
  "ts": "2025-08-22T10:30:11Z",
  "workflowId": "wf_abc123",
  "activity": "call_model",
  "attempt": 2,
  "latency_ms": 853,
  "token_in": 1250,
  "token_out": 340,
  "status": "success"
}
```

### Key Features
    ✅ Strict JSON Output - LLM responses are validated and structured
    ✅ Docker Containerized - Easy setup and deployment
    ✅ Temporal Orchestration - Reliable workflow execution with retries
    ✅ REST API - Standard HTTP endpoints for integration
    ✅ Structured Logging - JSON logs with metrics and attempt tracking
    ✅ Artifact Management - Browse and retrieve stored workflow data [todo]

## Workflow Process 
    1. Trigger: POST request starts a Temporal workflow
    2. LLM Call: Worker calls Gemini API with structured prompt
    3. JSON Validation: Response is parsed and validated
    4. Artifact Storage: Metadata and logs are stored 
    5. Result Return: Structured JSON result is available via API

