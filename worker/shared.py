from dataclasses import dataclass
from typing import List

INFORMATION_TASK_QUEUE_NAME = "INFORMATION_TASK_QUEUE"
@dataclass
class InvoiceData:
    context_input: List[str] 
    output:List[dict]
    fields_to_extract : list
    workflow_id: str

# class InvoiceData:
#     context_input: str 
#     fields_to_extract : list
#     output:dict
#     workflow_id: str