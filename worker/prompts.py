def get_prompt(context:str)->str:
    prompt = f"""
                You are provided with an invoice text. Your task is to extract the relevant entities and structure them into a JSON format. The expected entities include: 
                - INVOICE_NUMBER
                - DATE_OF_ISSUE (or DATE)
                - SERVICE_DATE (if applicable)
                - BILLED_TO (or ISSUED_TO)
                - ADDRESS
                - PHONE (if available)
                - EMAIL (if available)
                - ITEM_DESCRIPTION
                - QTY (or QUANTITY)
                - UNIT_PRICE (or PRICE)
                - AMOUNT (or TOTAL)
                - SUBTOTAL (if applicable)
                - TAX (if applicable)
                - TOTAL_AMOUNT (or GRAND_TOTAL)
                - BANK_NAME
                - ACCOUNT_NAME
                - ACCOUNT_NUMBER
                - PAYMENT_TERMS (if available)
                - PAYMENT_DATE (if applicable)

                Even if some entities have slightly different names or are missing, focus on extracting as much relevant information as possible and match them to the appropriate keys in the JSON format.
                here is the examples for more detailed understanding 
                
                
                {few_shot()}

                ### Instruction:
                Extract the relevant entities from the given input invoice text and structure them in JSON format.
                
                Input:
                {context}

                Response : 


                """ 
    return prompt

def few_shot()->str:

    few_shot_prompt = """
                you can follow below example, 

                <EXAMPLE>
                    INPUT: Cream and White Simple Minimalist Catering Services Invoice

                        INVOICE
                        Borcelle | Catering Services

                        DESCRIPTION QTY TOTAL

                        TOTAL $1000

                        ISSUED TO:

                        Amount  due $550

                        INVOICE NO:

                        #612345

                        UNIT PRICE

                        BANK DETAILS

                        Grilled Chicken 2 $200

                        Bruschetta 2 $200

                        Roasted Vegetables 2 $200

                        Tiramisu 2 $200

                        Helena Paquet
                        Borcelle #029 Client
                        hello@reallygreatsite.com

                        Total

                        Tax

                        $500
                        10%

                        Fresh Fruit Punch 2 $200

                        MARCH.06.2024

                        100

                        100

                        100

                        100

                        100

                        Borcelle Bank
                        Account Name: Avery Davis
                        Account No.: 123-456-7890
                        Pay by: 5 July 2025

                    OUTPUT:

                    {
                        'TOTAL_AMOUNT': '$1000',
                        'DUE_AMOUNT': '$550',
                        'INVOICE_NUMBER': '#612345',
                        'ITEM_DESCRIPTION': 'Grilled Chicken',
                        'QTY': '2',
                        'UNIT_PRICE': '$200',
                        'BANK_NAME': 'Borcelle Bank',
                        'ACCOUNT_NAME': 'Avery Davis',
                        'ACCOUNT_NUMBER': '123-456-7890',
                        'PAYMENT_DATE': '5 July 2025'
                    }

                </EXAMPLE>
        
        """
    return few_shot_prompt