#!/bin/bash

# Start services in background
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Test workflow trigger using test.json
echo "Testing workflow with test.json..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/workflows/trigger" \
  -H "Content-Type: application/json" \
  --data @tests/test.json)

echo "Trigger Response: $RESPONSE"

# Extract workflow_id from response
WORKFLOW_ID=$(echo $RESPONSE | grep -o '"workflow_id":"[^"]*' | cut -d'"' -f4)
echo "Workflow ID: $WORKFLOW_ID"

# Wait for workflow completion
echo "Waiting for workflow to complete..."
while true; do
  STATUS_RESPONSE=$(curl -s "http://localhost:8000/workflows/$WORKFLOW_ID/status")
  STATUS=$(echo $STATUS_RESPONSE | grep -o '"status":"[^"]*' | cut -d'"' -f4)
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "COMPLETED" ]; then
    break
  elif [ "$STATUS" = "FAILED" ]; then
    echo "Workflow failed"
    exit 1
  elif [ "$STATUS" = "CANCELED" ]; then
    echo "Workflow CANCELED"
    exit 1
  fi
  
  sleep 2
done

# Get workflow result
echo "Getting workflow result..."
RESULT=$(curl -s "http://localhost:8000/workflows/$WORKFLOW_ID/result")
echo "Result: $RESULT"
echo "✅ Test complete."