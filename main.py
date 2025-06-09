from flask import Flask, request, jsonify
import PyPDF2
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS
from dotenv import load_dotenv
import os
load_dotenv()
uri = os.getenv("MONGO_URI")
hf_token = os.getenv("HF_TOKEN")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["usercredentials"]
collection = db["login"]
status = db["status"]
registeration = db["registeration_number"]
app = Flask(__name__)
CORS(app)
 
# Load models globally
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
model, tokenizer = None, None

# Load Mistral model (Initialize it once)
def load_mistral(local_dir="mistral-7b-model", _token=None):
    global model, tokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    if os.path.exists(local_dir):
        model = AutoModelForCausalLM.from_pretrained(local_dir)
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=_token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=_token)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
    return model, tokenizer

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

# Create vector database
def create_vector_database(text, urls):
    sentences = text.split('.')
    sentence_embeddings = sentence_model.encode(sentences)
    url_embeddings = sentence_model.encode(urls)
    all_embeddings = np.vstack((sentence_embeddings, url_embeddings))
    all_items = sentences + urls
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)
    return index, all_items

# RAG Query function
def rag_query(query, index, sentences, model, tokenizer):
    query_embedding = sentence_model.encode([query])
    D, I = index.search(query_embedding, 3)
    relevant_text = ". ".join([sentences[i] for i in I[0]])
    input_text = f"Context: {relevant_text}\n\nQuery: {query}"
    inputs = tokenizer(input_text, max_length=96, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100, num_beams=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Initialize FAISS index and sentences
index = None
sentences = []

# 1. Query the database
@app.route('/query', methods=['POST'])
def query_database():
    global index, sentences
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if index is None or not sentences:
        return jsonify({"error": "PDF has not been processed properly."}), 500
    
    # Query the database using the RAG model
    response = rag_query(query, index, sentences, model, tokenizer)
    return jsonify({"response": response})

@app.route('/signup', methods = ['POST'])
def signup():
    username = request.json.get('username')
    password = request.json.get('password')
    result = collection.insert_one(request.json)
    return jsonify({"message " : "data added successfully"}),200

@app.route('/login',methods = ['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    data = collection.find_one({"username" : username})
    if (data):
        if(username == data["username"] and password == data["password"]):
            temp = registeration.find({"username" : username})
            temp = [x['registeration_number'] for x in temp]
            return jsonify({"message " : str(temp)}),200
        else: 
            return jsonify({"message " : "bad credential"}),401
    else:
         return jsonify({"message " : "bad credential"}),401
    
@app.route('/get-status',methods = ['GET'])
def get_status():
    registeration_number = request.args.get('registeration_number')
    print(registeration_number)
    status_var = status.find_one({"registeration_number": str(registeration_number)})
    return jsonify({"message" : str(status_var)})
# Static PDF processing at startup
def setup_pdf():
    global index, sentences
    # Static PDF path
    pdf_path = "urlcpgrams.pdf"  # Make sure this PDF is in the project directory
    pdf_text = extract_text_from_pdf(pdf_path)

    # Example URLs (adjust or make dynamic if needed)
    urls = [
        'https://pgportal.gov.in/Signin', 'https://pgportal.gov.in/Status', 
        'https://pgportal.gov.in/Home/Faq', 'https://pgportal.gov.in/Sitemap',
        'https://pgportal.gov.in/ContactUs', 'https://www.facebook.com/DARPGIndia/',
        'https://www.youtube.com/@darpg5380', 'https://pgportal.gov.in/Aboutus',
        'https://pgportal.gov.in/Home/LodgeGrievance', 'https://pgportal.gov.in/pension/',
        'https://pgportal.gov.in/Reminder', 'https://pgportal.gov.in/status/Index/F2A735FA35FCA11528479B9308348C17',
        'https://pgportal.gov.in/Home/NodalAuthorityForAppeal', 'https://pgportal.gov.in/Home/ProcessFlow'
    ]

    # Create the FAISS index
    index, sentences = create_vector_database(pdf_text, urls)

if __name__ == "__main__":
    # Hugging Face token
    model, tokenizer = load_mistral(_token=hf_token)
    
    # Process the static PDF once on server start
    setup_pdf()
    
    # Run the Flask server
    app.run(host='0.0.0.0', port=8080)
