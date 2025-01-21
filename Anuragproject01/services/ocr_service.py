import os
import fitz  # PyMuPDF for PDFs
import tempfile
from PIL import Image
import pytesseract
import docx
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import logging
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the language model
llm = ChatGroq(
    groq_api_key='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',  # Replace with your actual API key
    temperature=0.2,
    max_tokens=3000,
    model_kwargs={"top_p": 1}
)

# Define the chat prompt template
template = """
<|context|>
You are a professional data analyst. Analyze the provided report text and additional data. Provide:
1. Key insights and patterns from the report and the past data.
2. Recommendations based on the report and the trends in the past data.
3. Highlight any important observations or trends.
4. Provide a summary of the numeric values mentioned in the report data and compare with additional data.
</s>
<|user|>
Report data: 
{report_text}

Additional Data:
{additional_data}
</s>
<|assistant|>
"""

prompt = ChatPromptTemplate.from_template(template)


def fetch_last_entries_mongo():
    try:
        # MongoDB URI for local connection
        client = MongoClient('mongodb://localhost:27017/')
        db = client['healthcare']  # Database name
        collection = db['health_data']  # Collection name
        
        # Fetch the last 5 entries from the collection
        entries = collection.find().sort("_id", -1).limit(5)
        return list(entries)
    except Exception as e:
        logging.error(f"MongoDB error: {e}")
        return []
# Function to pass text to LLM for analysis
def analyze_text_with_llm(report_text, additional_data):
    try:
        formatted_data = '\n'.join([str(row) for row in additional_data]) if additional_data else "No additional data available."
        # Format the prompt
        formatted_prompt = prompt.format(report_text=report_text, additional_data=formatted_data)
        # Directly pass formatted prompt to LLM (without using to_input_dict)
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        logging.error(f"Error while processing with LLM: {e}")
        return "Error: Unable to analyze the report."

# Function to process image files
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Function to process PDF files using fitz
def extract_text_from_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)  # Save the uploaded file to the temporary file
        pdf_file = fitz.open(temp_file.name)
        text = ""
        for page_num in range(pdf_file.page_count):
            page = pdf_file.load_page(page_num)
            text += page.get_text()
    return text

# Function to process Word documents
def extract_text_from_word(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Main function to analyze report based on file type
def analyze_report(file):
    file_extension = file.filename.split('.')[-1].lower()

    # Extract text based on file type
    if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
        extracted_text = extract_text_from_image(file)
    elif file_extension == 'pdf':
        extracted_text = extract_text_from_pdf(file)
    elif file_extension in ['doc', 'docx']:
        extracted_text = extract_text_from_word(file)
    else:
        return "Unsupported file type."

    # Fetch additional data from the database
    additional_data = fetch_last_entries_mongo()

    # Analyze extracted text with LLM
    return analyze_text_with_llm(extracted_text, additional_data)
