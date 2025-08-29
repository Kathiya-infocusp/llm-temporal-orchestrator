import os

import json


from dotenv import load_dotenv

import google.generativeai as genai
from google.api_core import exceptions
from typing import Optional

from worker.utils import validate_extracted_data 
from worker.utils import simplify_response 
from worker.prompts import get_prompt

load_dotenv() 


class gemini:
    def __init__(self,name:str = "gemini-1.5-flash"):
        self.name = name 
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set.")
            genai.configure(api_key=api_key)
        except ValueError as e:
            print(f"Error: {e}")
            print("Please set your GOOGLE_API_KEY environment variable.")
            exit() 

        self.model_reponse=None
        
    def load_input(self,fields_to_extract:list) -> dict:
        """
        validate request, normalize context/schema/questions
        """
        
        if not fields_to_extract : 
            return {"status":"failed","error" : "fields_to_extract is none", "details": ""}
        properties = {field: {"type": "string"} for field in fields_to_extract}
        self.json_schema = {
            "type": "object",
            "properties": properties,
            "required": fields_to_extract
        }

        
        return {"status":"success","error" : "", "details": ""}


    def construct_prompt(self,context:str)->dict:
        """
        build a single prompt with strict instructions
        """

        self.prompt = get_prompt(context)
        return {"status":"success","error" : "", "details": ""}

    def call_model(self)->dict:
        try:
            self.model = genai.GenerativeModel(
                        model_name=self.name,
                        generation_config={
                            "response_mime_type": "application/json",
                            "response_schema": self.json_schema
                        }
                    )
            

            response = self.model.generate_content(self.prompt)
            self.model_reponse = response
            return {"status":"success","error" : "", "details": ""}
        except exceptions.GoogleAPICallError as e:
            print(f"An error occurred with the Google API: {e}")
            return {"status":"failed","error": "API call failed", "details": str(e)}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"status":"failed","error": "An unexpected error occurred", "details": str(e)}

    def parse_and_validate(self, context: str, required_fields: list) ->dict:
        try:
            extracted_json = json.loads(self.model_reponse.text)
            
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from model response.")
            return {"status":"failed","error": "Invalid JSON response from model", "details": str(e)}

        validation_errors = validate_extracted_data(extracted_json, context, required_fields)

        if validation_errors:
            print(f"Failed in validation criteria from model response.")
            return {"status":"failed","error": "Failed in validation criteria from model response", "details": [str(e) for e in validation_errors]} 
        else:      
            return extracted_json


    def persist_artifacts (self):
        pass 

    def finalize(self):
        pass 
