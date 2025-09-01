import os

import json
import pickle 

from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai
from google.api_core import exceptions
from typing import Optional

from worker.utils import validate_extracted_data
from worker.utils import normalize_text
from worker.utils import simplify_response 
from worker.prompts import get_batched_prompt
from worker.shared import InvoiceData

load_dotenv() 

def save_json_artifact(data: dict, output_dir: str, filename: str):
    """
    Saves a dictionary as a JSON file in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    return file_path

class gemini:
    def __init__(self,name:str = "gemini-2.5-pro"):
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
        
    def load_input(self,data:InvoiceData) -> dict:
        """
        validate request, normalize context/schema/
        """
        
        if not data : 
            return {"status":"failed","error" : "data is none", "details": ""}
        # properties = {field: {"type": "string"} for field in fields_to_extract}
        # self.json_schema = {
        #     "type": "object",
        #     "properties": properties,
        #     "required": fields_to_extract
        # }

        self.inovices = [normalize_text(_) for _ in data.context_input]
        self.output = None
        if data.output:
            self.output = data.output 
        self.required_fields = data.fields_to_extract

        return {"status":"success","error" : "", "details": ""}


    def construct_prompt(self)->dict:
        """
        build a single prompt with strict instructions
        """

        self.prompt = get_batched_prompt(self.inovices)
        return {"status":"success","error" : "", "details": ""}

    def call_model(self)->dict:
        try:
            self.model = genai.GenerativeModel(
                        model_name=self.name,
                        generation_config={
                            "response_mime_type": "application/json",
                            # "response_schema": self.json_schema  ## TODO add schema for list of json objects 
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

    def parse_and_validate(self) ->dict:
        try:
            extracted_responses = json.loads(self.model_reponse.text)
            # extracted_responses = [ json.loads(_) for _ in extracted_json]
            
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from model response.")
            return {"status":"failed","error": "Invalid JSON response from model", "details": str(e)}

        self.validated_response = []
        for i in range(len(extracted_responses)):
            validation_error = validate_extracted_data(extracted_responses[i], self.inovices[i], self.required_fields[i])
            if validation_error:
                print(f"Failed in validation criteria from model response for index : ",[str(e) for e in validation_error])
            self.validated_response.append(extracted_responses[i])

        
        if not self.validated_response:
            print(f"Failed in validation criteria from model response.")
            return {"status":"failed","error": "Failed in validation criteria from model response", "details":""} 
        else:      
            return {"status":"success","error" : "", "details": ""}


    def persist_artifact(self,path:Path):
        try:
            input_data = {
                'inovices' : self.inovices,
                'required_fields' : self.required_fields,
            }

            if self.output:
                input_data['ground_truth'] = self.output

            save_json_artifact(input_data,path,'input_data.json')
            save_json_artifact(self.validated_response,path,'extratced_model_response.json')
            # save_json_artifact(self.model_reponse.text,path,'model_response.json')

            file_path = os.path.join(path, 'final_prompt.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.prompt) 
            
            return {"status":"success","error" : "", "details": ""}
        except Exception as e:
            print(f"Failed to save model artifacts.")
            return {"status":"failed","error": "Failed to save artifacts", "details": str(e)}

    def finalize(self):
        """
        add evalution and summery 
        """
        #TODO add evalution script and share the results on final call.
        return { "results " : self.validated_response}

        
