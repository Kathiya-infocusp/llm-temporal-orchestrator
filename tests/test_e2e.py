import pytest
import requests
import json
import time
import os
from pathlib import Path

class TestE2E:
    BASE_URL = "http://localhost:8000"
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Wait for services to be ready"""
        print("Waiting for services to be ready...")
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.BASE_URL}/docs", timeout=5)
                if response.status_code == 200:
                    print(f"Services ready after {i+1} attempts")
                    break
                print(f"Service check {i+1}/{max_retries}: {response.status_code}")
            except Exception as e:
                print(f"Service check {i+1}/{max_retries}: {e}")
                time.sleep(2)
        else:
            pytest.fail("Services not ready after 60 seconds")
    
    def test_full_workflow_success(self):
        """Test complete workflow from trigger to result"""
        # Load test data
        with open("tests/test.json") as f:
            test_data = json.load(f)
        
        # 1. Trigger workflow
        print(f"Triggering workflow with data: {json.dumps(test_data, indent=2)[:200]}...")
        response = requests.post(f"{self.BASE_URL}/workflows/trigger", json=test_data)
        
        if response.status_code != 200:
            print(f"Failed to trigger workflow: {response.status_code} - {response.text}")
            pytest.fail(f"Workflow trigger failed: {response.status_code} - {response.text}")
        
        workflow_id = response.json()["workflow_id"]
        print(f"Started workflow: {workflow_id}")
        assert workflow_id.startswith("workflow-")
        
        # 2. Poll status until completion
        max_wait = 180  # 3 minutes
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait:
            try:
                status_response = requests.get(f"{self.BASE_URL}/workflows/{workflow_id}/status")
                if status_response.status_code != 200:
                    print(f"Status check failed: {status_response.status_code} - {status_response.text}")
                    time.sleep(2)
                    continue
                    
                status = status_response.json()["status"]
                
                if status != last_status:
                    print(f"Workflow status: {status} (elapsed: {time.time() - start_time:.1f}s)")
                    last_status = status
                
                if status == "COMPLETED":
                    print(f"Workflow completed in {time.time() - start_time:.1f}s")
                    break
                elif status in ["FAILED", "CANCELED"]:
                    # Get result to see error details
                    result_response = requests.get(f"{self.BASE_URL}/workflows/{workflow_id}/result")
                    error_details = result_response.json().get("error", "No error details")
                    pytest.fail(f"Workflow failed with status: {status}. Error: {error_details}")
                
            except requests.RequestException as e:
                print(f"Request error during status check: {e}")
            
            time.sleep(3)  # Increased polling interval
        else:
            print(f"Workflow timed out after {max_wait}s. Last status: {last_status}")
            # Try to get any available result/error info
            try:
                result_response = requests.get(f"{self.BASE_URL}/workflows/{workflow_id}/result")
                if result_response.status_code == 200:
                    result_info = result_response.json()
                    print(f"Final result info: {result_info}")
            except:
                pass
            pytest.fail(f"Workflow did not complete within {max_wait}s timeout")
        
        # 3. Get result
        result_response = requests.get(f"{self.BASE_URL}/workflows/{workflow_id}/result")
        assert result_response.status_code == 200
        result = result_response.json()
        
        print(f"Final result: {json.dumps(result, indent=2)[:500]}...")
        
        assert result["status"] == "COMPLETED"
        assert result["result"] is not None
        assert "predictions" in result["result"]
        
        # 4. Verify artifacts created
        runs_dir = Path("runs") / workflow_id
        print(f"Checking artifacts in: {runs_dir}")
        assert runs_dir.exists(), f"Runs directory {runs_dir} does not exist"
        assert (runs_dir / "workflow.log").exists(), "workflow.log not found"
        assert (runs_dir / "input_data.json").exists(), "input_data.json not found"
    
    def test_invalid_input_handling(self):
        """Test workflow with invalid input data"""
        invalid_data = {"context_input": "invalid", "output": "invalid"}
        
        response = requests.post(f"{self.BASE_URL}/workflows/trigger", json=invalid_data)
        assert response.status_code == 500
    
    def test_nonexistent_workflow_status(self):
        """Test status check for non-existent workflow"""
        response = requests.get(f"{self.BASE_URL}/workflows/fake-id/status")
        assert response.status_code == 500