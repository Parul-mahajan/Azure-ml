from azure.identity import ManagedIdentityCredential, DefaultAzureCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azureml.core import Run
import re
from datetime import datetime
import pdfplumber
import os
import pandas as pd
import openai
import chardet
from decimal import Decimal, InvalidOperation
import glob
import azure.ai.formrecognizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import numpy as np
np.int = np.int32
np.float = np.float64
np.bool = np.bool_
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import fitz
# import PyMuPDF
from dateutil import parser
from datetime import datetime
import pytz
import ast
import multiprocessing
import os
import openai
from openai import AzureOpenAI
import langdetect
from langdetect import detect
import ast


LOCAL_RUN = 'N'

pd.set_option('mode.chained_assignment', None)

if LOCAL_RUN == 'N':
    # Get the experiment run contexty
    run = Run.get_context()
    ws = run.experiment.workspace
else:
    # Load the workspace from the saved config file
    ws = Workspace.from_config()


print('Ready to work with {}'.format(ws.name))

keyvault = ws.get_default_keyvault()

identity_client_id = keyvault.get_secret("clientid")

credential = ManagedIdentityCredential(client_id=identity_client_id)
print("credential", credential)

openai.api_type = "azure"
openai.api_base = "https://openai-finops-np-002.openai.azure.com/"
openai.api_version = "2024-02-15-preview"
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
 
client = AzureOpenAI(
  azure_endpoint = openai.api_base, 
  azure_ad_token_provider=token_provider, 
  api_version=openai.api_version
) 

start_time = time.time()

def get_completion(prompt):
    response = client.chat.completions.create(
        model="openainpfinos02",
        seed=42,
        response_format={ "type": "json_object" },
        messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content":prompt}])
    response = response.choices[0].message.content
    return response  


def form_reading_pdf(filename, endpoint, credential):
    print("inside form recognizer")
    client = DocumentAnalysisClient(endpoint, credential = credential)
 
    # Use the custom model to extract information from the document
    with open(filename, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", f.read())
        result = poller.result()
 
    output = {}
    page_number = 1  # Initialize page number
    for page in result.pages:
        text = ""
        for line in page.lines:
            text += line.content + "\n"
        output['Page number ' + str(page_number)] = text.strip()
        page_number += 1  # Increment page number
 
    return json.dumps(output)


def form_convert_dict_to_list_of_dicts(input_string):
    input_dict = ast.literal_eval(input_string)
    list_of_dicts = []
    for key, value in input_dict.items():
        list_of_dicts.append({key: value})
    return list_of_dicts

def convert_dict_to_list_of_dicts(input_dict):
    # input_dict = ast.literal_eval(input_string)
    list_of_dicts = []
    for key, value in input_dict.items():
        list_of_dicts.append({key: value})
    return list_of_dicts

def reading_pdf(filename, endpoint, credential):
    print("inside reading pdf func",filename)
    doc = fitz.open(filename)
    page = doc[0]
    if len(page.get_text())>100:
        print("********pdf is editable*********",len(page.get_text()))
        
    # print(len(doc))
        final_dict = {}
        for i in range(len(doc)):
            page = doc[i]  # The third page is at index 2 
            page_content = page.get_text()
            page_number_key = f"Page number {i+1}"
            final_dict[page_number_key] = page_content
            final_dict_1 = convert_dict_to_list_of_dicts(final_dict)
        return final_dict_1

    else:
        print("**************pdf is not editable*************")
        final_dict = form_reading_pdf(filename, endpoint, credential)
        final_dict_1 = form_convert_dict_to_list_of_dicts(final_dict)
        return final_dict_1
    
model1 = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

## Compute embeddings
def get_embeddings(text, output_path_emb):
    print("inside embeddings")
    embeddings = model1.encode(text)
    with open(output_path_emb, 'wb') as f:
        pickle.dump(embeddings, f)
    # return embeddings
    
def load_embeddings_from_pickle(embeddings_path):
    print("Inside load embedding func")
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def save_list_to_file(pdf_text, output_path):
    print("inside save_list_to_file")
    pdf_text = str(pdf_text)
    with open(output_path, "w") as file:
        file.write(pdf_text)
        
def read_list_from_file(filename):
    print("inside read_list_from_file")
    with open(filename, 'r') as file:
        return file.read()

def find_closely_related_chunks(query, embeddings, text, threshold=0.0, top_k=15):
    similarity_scores = cosine_similarity([query], embeddings)[0]
    above_threshold_indices = np.where(similarity_scores > threshold)[0]
    top_k_indices = above_threshold_indices[np.argsort(similarity_scores[above_threshold_indices])[::-1][:top_k]]
    
    print("Top 3 similarity scores:",format(top_k))
    for idx in top_k_indices:
        print("Index {}: {}".format(idx, similarity_scores[idx]))
    
    closely_related_chunks = []
    for idx in top_k_indices:
        chunk_key = list(text[idx].keys())[0]
        similarity_score = similarity_scores[idx]
        closely_related_chunks.append({f"{chunk_key}, similarity_score: {similarity_score:.8f}": text[idx][chunk_key]})
    
    return closely_related_chunks

def create_chunks(list_of_dicts,chunk_size=1):
    # print("list_of_dicts-------------",len(list_of_dicts))
    # print("list_of_dicts-------------",list_of_dicts)
    chunks = []
    for i in range(0, len(list_of_dicts), chunk_size):
        chunk = list_of_dicts[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def summary1(contract):
    prompt = f"""
    You will be provided a contract details {contract}. You need to compose a brief description of the contract, specifying the parties involved as the purchaser, the supplier, and affliates and clarify their roles. Display the response in JSON format."""
    response = get_completion(prompt)
    # print('response---------------------------------------',response)
    return response
 
def extract_text_from_image(image_bytes, endpoint, credential):
    # Initialize DocumentAnalysisClient
    client = DocumentAnalysisClient(endpoint, credential = credential)
 
    # Call Azure Form Recognizer to extract text from the image
    poller = client.begin_analyze_document("prebuilt-layout", image_bytes)
    result = poller.result()
 
    # Extract text from the result
    text = ""
    for page in result.pages:
        for line in page.lines:
            text += line.content + "\n"
 
    return text.strip()
 
def extract_text_from_pdf(pdf_path, endpoint, credential):
    print("Extracting text from PDF...")
    output = []
    output1 = []
    # Initialize DocumentAnalysisClient
    client = DocumentAnalysisClient(endpoint, credential = credential)
 
    # Open the PDF file
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            if image_list:
                print('image')
                # Extract text from the image using Azure Form Recognizer
                image_bytes = page.get_pixmap().tobytes()
                extracted_text = extract_text_from_image(image_bytes, endpoint, credential)
                output1.append({f"Page number {page_num+1}": extracted_text})
            else:
                text = page.get_text()            
                # print('editable')
                output.append({f"Page number {page_num+1}": text.strip()})
 
    return output1

def format_response(text, response):
    
    # Step 1: Extract clause reference value
    clause_reference = next((value for key, value in response.items() if 'clause' in key.lower()), None)

    # Step 2: Automatically remove keys containing the word "clause"
    keys_to_remove = [key for key in response if 'clause' in key.lower()]
    for key in keys_to_remove:
        response.pop(key, None)
    # print('updated response---------------',response)

    # Step 3: Extract page number and similarity score
    page_number = ""
    similarity_score = ""
    for item in text:
        for key in item.keys():
            match = re.search(r'Page number (\d+), similarity_score: (\d+\.\d+)', key)
            if match:
                page_number = match.group(1)
                similarity_score = float(match.group(2))

    # Step 4: Determine rank
    if similarity_score > 0.3:
        rank = "Rank High"
    elif 0.2 <= similarity_score <= 0.3:
        rank = "Rank Medium"
    else:
        rank = "Rank Low"

    # Step 4: Update response dictionary
    for key in response:
        response[key] = f"{response[key]}:{clause_reference}:{rank}:Page {page_number}:score {similarity_score:.2f}"
    
    return response
    # print('new response------------------------------',response)
    
    
def map_keys_by_order(data, new_keys):
    if len(data) != len(new_keys):
        raise ValueError("Number of new keys must match the number of keys in the dictionary.")

    updated_dict = {}
    for i, (key, value) in enumerate(data.items()):
        updated_dict[new_keys[i]] = value

    return updated_dict
    
def is_numeric(val):
    try:
        int(val)
        return True
    except ValueError:
        return False

def process_payment_terms(data):
    payment_terms = data.get('PaymentTerms', '').lower()
    payment_terms_in_days = data.get('PaymentTermsInDays', '')

    if payment_terms == 'numeric':
        if is_numeric(payment_terms_in_days):
            return data
        else:
            for key in data:
                data[key] = 'na'
            return data
    else:
        return data
    
    
def process_payment_clause(data):
    payment_terms_ref = data.get('Payment Terms_Clause Reference', '').lower()
    payment_terms = data.get('PaymentTerms', '').lower()
    payment_terms_in_days = data.get('PaymentTermsInDays', '').lower()

    if payment_terms_ref != 'na' and payment_terms == 'na' and payment_terms_in_days == 'na':
        for key in data:
            data[key] = 'na'

    return data

def process_dict(d, keys, valid_values):

    """

    Converts dictionary values to uppercase and replaces them if they match valid_values.
 
    Args:

        d (dict): The dictionary containing the values to be processed.

        valid_values (list): List of valid values.
 
    Returns:

        dict: The modified dictionary.

    """

    # Convert valid_values to lowercase
    lower_valid_values = [value.lower() for value in valid_values]
 
    # Convert dictionary values to lowercase and replace if valid

    for key, value in d.items():
        if key == keys:
            # print(keys)
            if isinstance(value, str):
                # print(value)
                lower_value = value.lower()
                # print("lower_value", lower_value)
                if lower_value in lower_valid_values:
                    d[key] = valid_values[lower_valid_values.index(lower_value)]
                    # print(d)
                else:
                    d[key] = 'na'
 
    return d

def convert_date_format(response, key):
    if key in response:
        value = response[key]
        if not isinstance(value, str):
            response[key] = 'na'
            return response
        try:
            parsed_date = parser.parse(value)
            formatted_date = parsed_date.strftime('%m/%d/%Y')
            response[key] = formatted_date
        except ValueError:
            response[key] = 'na'
    return response

def convert_pipe_to_space(dictionary):
    # Iterate over each key-value pair in the dictionary
    for key, value in dictionary.items():
        # If the value is a string and contains '|'
        if isinstance(value, str) and '|' in value:
            # Replace '|' with space
            dictionary[key] = value.replace('|', ' ')
    return dictionary

def process_clause_section(response):
    for key in response.keys():
        if 'clause' in key.lower():
            response[key] = ''.join(ch if ch.isalnum() or ch in ['.', '-'] else ' ' for ch in response[key])
    return response

def remove_no_na_response(response):
    # Get the first key of the dictionary
    first_key = list(response.keys())[0]

    if response[first_key].lower() == 'no':
        for key in response.keys():
            if 'clause' in key.lower():
                if response[key].lower() == 'na':
                    for key in response.keys():
                        response[key] = 'na'
                        
    return response

def get_details_Payment_terms(chunks):
     
    l = []
    for text in chunks:
        keywords = ['Payment terms','Payment Terms in days','Payment Terms_Clause Reference']
        

        output_format = """
            {"<Payment terms>":"<Identified answer>",
            "<Payment Terms in days>":"<Identified answer>",           
            "<Payment Terms_Clause Reference>":"<Identified answer>"}"""
                
        
        prompt = f"""
        You are a master service agreement (MSA) details provider AI assistance.\n
        Your task is you will be provided a list of keywords {keywords} you need to extract the relevant answer from the MSA Agreement.\n
        You will be provided a top keyword matching pages from the MSA Agreement delimited by triple backticks.\n
        Do not provide extra explantion to your answer. If you are unable to extract the answer, then leave it as 'na'. Do strictly follow the below instruction strictly.\n
        
        Instruction:\n
 
        Step1: Understand the contract and locate the 'Payment terms'. Based on the context, set 'payment terms' to either 'Numeric' or if statement of work is mentioned, then set it to 'As specified by SOW'. Other than these two options, don't provide response to payment terms..\n
        Step2: If 'Payment terms' is 'Numeric' then extract the payment terms in days value and set it to 'Payment Terms in days' elif 'Payment terms' is not 'numeric' then set 'Payment Terms in days' to 'na'.\n 
        Step3: If the value of 'payment terms' is either **Numeric** or **As specified by SOW**, you extract the corresponding clause/subclause/Appendix/Execution/Annex/Exhibit and assign it to 'Payment Terms_Clause Reference' where the payment terms are mentioned. If you unable to extract 'Payment Terms_Clause Reference' then leave it as 'na'.
        'Payment terms_Clause Reference' this value should be in the form of: 'Sec <section number>'.
        Step4: Display the response strictly in a JSON format:{output_format}. 

        MSA Agreement : ```{text}```
        keywords: {keywords}
        """
        for _ in range(1):
            
            try:

                response = get_completion(prompt)
                response = ast.literal_eval(response)
                print('response------------------------------', response)

                #changing the key name in response
                new_keys= ['PaymentTerms','PaymentTermsInDays','Payment Terms_Clause Reference']
                response = map_keys_by_order(response,new_keys)
                response = convert_pipe_to_space(response)
                response = process_clause_section(response)
                #validation
                valid_values = ['Numeric', 'As specified in SOW', 'na']
                response = process_dict(response, 'PaymentTerms', valid_values)

                response = process_payment_terms(response)  #checking numeric - days in integer 
                print('response checking numeric int:',response)

                response = process_payment_clause(response)
                print('response checking clause :',response)
                
                response = remove_no_na_response(response)
                print('removing No and na response-------------',response)

                #If all values are na then not considering
                if all(value.lower() != 'na' for value in response.values()):
                    print("Not all values are 'na'. Proceeding to the next text.")
                    # continue

                    response = format_response(text,response)
                    print('formated response------------------------------',response)

                    l.append(response)
                break
            except Exception as e:
                if '429' in str(e):  # If error is 429
                    print("429 error encountered. Waiting for 4 seconds before retrying.")
                    time.sleep(4)  # Wait for 4 seconds before retrying
                else:
                    print(f"Error processing text: {e}")
                    break  # If error is not 429, break the retry loop and continue with the next text
    return l
        
    
