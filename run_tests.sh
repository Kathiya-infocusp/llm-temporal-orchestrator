#!/bin/bash

echo "Starting End-to-End Test Suite"

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services
echo "Waiting for services to be ready..."
while ! curl -s http://localhost:8000/docs > /dev/null 2>&1; do
  echo "Services not ready, waiting..."
  sleep 2
done
echo "Services ready!"

# Run different test suites
echo "Running test suites..."

# Unit tests
echo "Running unit tests..."
python -m pytest tests/test_llms.py -v

# Integration tests  
echo "Running integration tests..."
python -m pytest tests/test_e2e.py -v

# Cleanup
echo "Cleaning up..."
docker-compose down

echo "All tests completed!"