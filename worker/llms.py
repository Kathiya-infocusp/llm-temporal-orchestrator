import os
import json
import pickle 
import time

from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions

from worker import utils
from worker.shared import InvoiceData
from worker.prompts import retry_prompt
from worker.prompts import get_batched_prompt



load_dotenv() 

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
        self.retry_prompt=""
        self.evalution_result=None
        self.validated_response=None
        
    def load_input(self,data:InvoiceData) -> dict:
        """
        validate request, normalize context/schema/
        """
        
        #TODO add request validations 
        if not data : 
            return {"status":"failed","error" : "data is none", "details": ""}
        
        # properties = {field: {"type": "string"} for field in fields_to_extract}
        # self.json_schema = {
        #     "type": "object",
        #     "properties": properties,
        #     "required": fields_to_extract
        # }

        self.inovices = [utils.normalize_text(_) for _ in data.context_input]
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
                        }
                    )
            
            start_time = time.time()
            response = self.model.generate_content(self.prompt)
            end_time = time.time()
            self.model_reponse = response.text
            self.latency = end_time - start_time 
            self.metadata = {
                "prompt_token_count":response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
                "total_token_count": response.usage_metadata.total_token_count,
            }
            return {"status":"success","error" : "", "details": ""}
        except exceptions.GoogleAPICallError as e:
            return {"status":"failed","error": "API call failed", "details": str(e)}
        except Exception as e:
            return {"status":"failed","error": "An unexpected error occurred", "details": str(e)}

    def parse_and_validate(self) ->dict:
        try:
            extracted_responses = json.loads(self.model_reponse)
            if type(extracted_responses[0]) == str:
                extracted_responses = [ json.loads(_) for _ in extracted_responses]
            
        except json.JSONDecodeError as e:
            return {"status":"failed","error": "Invalid JSON response from model", "details": str(e)}

        try:
            self.validated_response = []
            self.error_response = []
            for i in range(len(extracted_responses)):
                validation_error = utils.validate_extracted_data(extracted_responses[i], self.inovices[i])
                if validation_error:
                    self.error_response.append((i,validation_error))
                
                self.validated_response.append(extracted_responses[i])
        except Exception as e:
            return {"status":"failed","error": "An unexpected error occurred", "details": str(e)}
        
        if self.error_response:
                return {"status":"failed","error": "Failed in validation criteria from model response", "details": self.error_response}
        else:      
            return {"status":"success","error" : "", "details": ""}



    def retry_model_call(self)->dict:
        """
        takes previously errorenous reponse, updates prompts with erros, and runs model 3 time to generate valid response,
        """

        if not self.error_response:
            return {"status":"success","error" : "", "details": ""}
        
        ids, erros = map(list, zip(*self.error_response))
        erroneous_context = [self.inovices[_] for _ in ids]

        for i in range(3):
            prompt = retry_prompt(erroneous_context, erros)

            response = self.model.generate_content(prompt)
            extracted_responses = json.loads(response.text)
            if extracted_responses and isinstance(extracted_responses[0], str):
                extracted_responses = [json.loads(_) for _ in extracted_responses]
            
            # Track successful validations
            successful_indices = []
            remaining_errors = []
            
            for j, extracted_response in enumerate(extracted_responses):
                validation_error = utils.validate_extracted_data(extracted_response, erroneous_context[j])
                if validation_error:
                    remaining_errors.append(validation_error)
                else:
                    self.validated_response[ids[j]] = extracted_response
                    successful_indices.append(j)

            # Remove successful items in reverse order
            for j in reversed(successful_indices):
                erroneous_context.pop(j)
                erros.pop(j)
                ids.pop(j)

            if not remaining_errors:
                self.retry_prompt += prompt
                return {"status":"success","error" : "", "details": ""}
            
            # Update error list for next iteration
            erros = remaining_errors
            
        return {"status":"failed","error": "Failed to generate correct response in 3 attempts", "details": ""}


    def finalize(self):
        """
        add evalution and summery 
        """

        self.evalution_result = utils.evaluate(self.output,self.validated_response)

        return { "evalution_resul " : self.evalution_result, "predictions" : self.validated_response}

        
    def persist_artifact(self,path:Path):
        try:
            input_data = {
                'inovices' : self.inovices,
                'required_fields' : self.required_fields,
            }

            if self.output:
                input_data['ground_truth'] = self.output

            utils.save_json_artifact(input_data,path,'input_data.json')

            file_path = os.path.join(path, 'final_prompt.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.prompt) 

            if self.model_reponse:
                response_artifacts = {
                    'model_response' : self.model_reponse,
                    'metadata' : self.metadata,
                    'latency' : self.latency
                }
                utils.save_json_artifact(response_artifacts,path,'model_response.json')


            if self.validated_response:
                utils.save_json_artifact(self.validated_response,path,'extratced_model_response.json')
            if self.evalution_result:
                utils.save_json_artifact(self.evalution_result,os.path.join(path,'eval'),'metrics.json')

            if self.retry_prompt:
                file_path = os.path.join(path, 'retry_prompt.txt')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.retry_prompt ) 
            
            return {"status":"success","error" : "", "details": ""}
        except Exception as e:
            return {"status":"failed","error": "Failed to save artifacts", "details": str(e)}