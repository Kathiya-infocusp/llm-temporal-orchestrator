from dataclasses import dataclass

INFORMATION_TASK_QUEUE_NAME = "INFORMATION_TASK_QUEUE"
@dataclass
class InvoiceData:
    context_input: str 
    fields_to_extract : list
    output:dict
    workflow_id: str