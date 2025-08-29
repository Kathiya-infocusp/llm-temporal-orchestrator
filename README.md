# LLM Service Orchestrated with Temporal (File-Based Storage)

A Python service that answers questions by orchestrating an LLM workflow with Temporal.

---

## Quick Start

### Set up environment:
```bash
cp .env.example .env
# Add your Gemini API key to .env
```
### Start all services:
```bash
docker-compose up --build
```
### Access the services:
 * **API**: http://localhost:8000
 * **Temporal UI**: http://localhost:8233
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
### Get workflow artifacts #TODO
```bash
curl "http://localhost:8000/artifacts/{workflow_id}"
```
### List all artifacts #TODO
```bash
curl "http://localhost:8000/artifacts"
```
### Key Features
    ✅ Strict JSON Output - LLM responses are validated and structured
    ✅ Docker Containerized - Easy setup and deployment
    ✅ Temporal Orchestration - Reliable workflow execution with retries
    ✅ REST API - Standard HTTP endpoints for integration
    ✅ Artifact Management - Browse and retrieve stored workflow data [todo]

## Workflow Process 
    1. Trigger: POST request starts a Temporal workflow
    2. LLM Call: Worker calls Gemini API with structured prompt
    3. JSON Validation: Response is parsed and validated
    4. Artifact Storage: Metadata and logs are stored [todo]
    5. Result Return: Structured JSON result is available via API

