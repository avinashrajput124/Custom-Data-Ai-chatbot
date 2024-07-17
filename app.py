import logging
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
import torch
import asyncio
import concurrent.futures
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cpu')

app = Flask(__name__)

def load_models():
    global tokenizer, base_model, llm_pipeline, qa_llm_resource
    checkpoint = "t5-small" 
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map=device,
        torch_dtype=torch.float32
    )
    llm_pipeline = create_llm_pipeline()
    qa_llm_resource = load_qa_llm_resource()

def create_llm_pipeline(detailed=False):
    max_length = 64 if not detailed else 150 
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=max_length,
        do_sample=False, 
        num_beams=2, 
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm
    
def load_qa_llm_resource():
    llm = create_llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings)
    return llm, db

def get_qa_llm(detailed=False):
    llm, db = qa_llm_resource
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})   
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer_sync(instruction, detailed=False):
    qa = get_qa_llm(detailed=detailed)
    start_time = time.time()
    generated_text = qa(instruction)
    logger.info(f"Generated answer: {generated_text}")
    logger.info(f"Time taken for QA inference: {time.time() - start_time} seconds")
    result_text = generated_text['result']
    
    keywords = [
        "context", 
        "not provided", 
        "not provide", 
        "don't have information",
        "not provide information",
        "unable to find",
        "no data"
    ]
    if any(keyword in result_text.lower() for keyword in keywords):
        answer = f"I apologize, but I do not have information related to {instruction}."
    else:
        answer = result_text
        if detailed and generated_text['source_documents']:
            source_document = generated_text['source_documents'][0]
            source_info = source_document.page_content
            answer = f"{answer}\n{source_info}"

    return answer

async def process_answer(instruction, detailed=False):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        answer = await loop.run_in_executor(pool, process_answer_sync, instruction, detailed)
    return answer

basic_intents = {
    "greet": ["hello", "hi", "hey"],
    "goodbye": ["bye", "goodbye", "see you"],
    "thanks": ["thank you", "thanks"],
    "help": ["help", "support"],
}

basic_responses = {
    "greet": "Hello! How can I assist you today?",
    "goodbye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! If you have any other questions, feel free to ask.",
    "help": "I can assist you with information and answer your questions. How can I help you?",
}
keywords_for_details = [
    "details",
    "more info",
    "elaborate",
    "in-depth",
    "explain",
    "full details",
    "comprehensive",
    "extended information",
    "detailed explanation",
    "complete answer",
    "thorough explanation",
    "deep dive"
]

def match_intent(user_input):
    for intent, keywords in basic_intents.items():
        for keyword in keywords:
            if keyword in user_input.lower():
                return intent
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
async def chat():
    user_input = request.json.get('message')
    detailed = any(keyword in user_input.lower() for keyword in keywords_for_details)
    intent = match_intent(user_input)
    if intent:
        answer = basic_responses[intent]
    else:
        answer = await process_answer(user_input, detailed=detailed)
    return jsonify({'response': answer})

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0')    
    # app.run(debug=True)
