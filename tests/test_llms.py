import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from google.api_core import exceptions
from worker.llms import  gemini
from worker.shared import InvoiceData

class TestGeminiModel(unittest.TestCase):
    
    @patch('worker.llms.genai.configure')
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'})
    def setUp(self, mock_configure):
        self.model = gemini()
        self.test_data = InvoiceData(
            context_input=["Test invoice content INVOICE_NUMBER : 123 \n TOTAL_AMOUNT : 100"],
            output=[{"INVOICE_NUMBER": "123", "TOTAL_AMOUNT": "100"}],
            fields_to_extract=[["INVOICE_NUMBER", "TOTAL_AMOUNT"]],
            workflow_id = "workflow-test"
        )
    
    # Success cases
    def test_load_input_success(self):
        result = self.model.load_input(self.test_data)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(self.model.inovices), 1)
    
    def test_construct_prompt_success(self):
        self.model.load_input(self.test_data)
        result = self.model.construct_prompt()
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(self.model.prompt)
    
    @patch('worker.llms.genai.GenerativeModel')
    def test_call_model_success(self, mock_model_class):
        mock_response = MagicMock()
        mock_response.text = '[{"INVOICE_NUMBER": "123", "TOTAL_AMOUNT": "100"}]'
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 100
        mock_response.usage_metadata.prompt_token_count = 200
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        self.model.load_input(self.test_data)
        self.model.construct_prompt()
        result = self.model.call_model()
        self.assertEqual(result["status"], "success")
    
    def test_parse_and_validate_success(self):
        self.model.load_input(self.test_data)
        mock_response = MagicMock()
        mock_response = """[{
            "INVOICE_NUMBER": "123", 
            "TOTAL_AMOUNT": "100"}]"""
        self.model.model_reponse = mock_response
        
        result = self.model.parse_and_validate()
        self.assertEqual(result["status"], "success")
    
    def test_finalize_success(self):
        self.model.load_input(self.test_data)
        self.model.validated_response = [{"INVOICE_NUMBER": "123"}]
        result = self.model.finalize()
        self.assertIn("predictions", result)
    
    # Failure cases
    def test_load_input_none_data(self):
        result = self.model.load_input(None)
        self.assertEqual(result["status"], "failed")
        self.assertIn("data is none", result["error"])
    
    @patch('worker.llms.genai.GenerativeModel')
    def test_call_model_api_error(self, mock_model_class):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = exceptions.GoogleAPICallError("API call failed")
        mock_model_class.return_value = mock_model
        
        self.model.load_input(self.test_data)
        self.model.construct_prompt()
        result = self.model.call_model()
        self.assertEqual(result["status"], "failed")
        self.assertIn("API call failed", result["error"])
    
    @patch('worker.llms.genai.GenerativeModel')
    def test_call_model_generic_exception(self, mock_model_class):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Generic error")
        mock_model_class.return_value = mock_model
        
        self.model.load_input(self.test_data)
        self.model.construct_prompt()
        result = self.model.call_model()
        self.assertEqual(result["status"], "failed")
        self.assertIn("An unexpected error occurred", result["error"])
    
    def test_parse_and_validate_invalid_json(self):
        mock_response = MagicMock()
        mock_response = "invalid json"
        self.model.model_reponse = mock_response
        
        result = self.model.parse_and_validate()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Invalid JSON response", result["error"])
    
    def test_parse_and_validate_no_valid_response(self):
        self.model.load_input(self.test_data)
        mock_response = MagicMock()
        mock_response = '[{"INVALID_FIELD": "value"}]'
        self.model.model_reponse = mock_response
        
        result = self.model.parse_and_validate()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Failed in validation criteria", result["error"])
    
    @patch('worker.utils.validate_extracted_data')
    @patch('worker.llms.genai.GenerativeModel')
    def test_retry_model_call_success(self, mock_model_class, mock_validate):
        mock_response = MagicMock()
        mock_response.text = '[{"INVOICE_NUMBER": "123", "TOTAL_AMOUNT": "100"}]'
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model 
        mock_validate.return_value = []
        
        self.model.load_input(self.test_data)
        self.model.model = mock_model
        self.model.error_response = [(0, ["test error"])]
        self.model.validated_response = [{}]
        
        result = self.model.retry_model_call()
        self.assertEqual(result["status"], "success")
    
    @patch('worker.utils.validate_extracted_data')
    @patch('worker.llms.genai.GenerativeModel')
    def test_retry_model_call_failure(self, mock_model_class, mock_validate):
        mock_response = MagicMock()
        mock_response.text = '[{"INVALID": "data"}]'
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        mock_validate.return_value = ["validation error"]  # Always return errors
        
        self.model.load_input(self.test_data)
        self.model.model = mock_model
        self.model.error_response = [(0, ["test error"])]
        self.model.validated_response = [{}]
        
        result = self.model.retry_model_call()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Failed to generate correct response", result["error"])
    
    # Exception handling cases
    @patch.dict('os.environ', {}, clear=True)
    def test_init_no_api_key(self):
        with self.assertRaises(SystemExit):
            gemini()
    
    @patch('worker.utils.save_json_artifact')
    def test_persist_artifact_success(self, mock_save):
        mock_save.return_value = "/tmp"
        self.model.load_input(self.test_data)
        self.model.validated_response = [{"test": "data"}]
        self.model.evalution_result = {"score": 1.0}
        self.model.model_reponse = MagicMock()
        self.model.model_reponse.text = "test response"
        self.model.metadata = {"tokens": 100}
        self.model.latency = 0.5
        self.model.prompt = "test prompt"
        
        result = self.model.persist_artifact(Path("/tmp"))
        self.assertEqual(result["status"], "success")
    
    @patch('worker.utils.save_json_artifact')
    def test_persist_artifact_failure(self, mock_save):
        mock_save.side_effect = Exception("Save error")
        
        result = self.model.persist_artifact(Path("/test"))
        self.assertEqual(result["status"], "failed")
        self.assertIn("Failed to save artifacts", result["error"])


if __name__ == "__main__":
    unittest.main()