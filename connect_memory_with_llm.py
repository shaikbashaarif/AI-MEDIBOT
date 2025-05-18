from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Step 1: Load environment variables
load_dotenv()

# Step 2: Setup LLM (Mistral-7B-Instruct with Hugging Face Inference)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "google/flan-t5-small"  # Your specified model

def load_llm(repo_id):
    llm = HuggingFaceEndpoint(
        endpoint_url=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        task="text-generation",
    )
    return llm

# Step 3: Setup custom prompt
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know — do not try to make up an answer.
Only use the information provided.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Step 4: Load vectorstore (FAISS)
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 5: Create Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)},
)

# Step 6: Query
user_query = input("Write your query here: ")
response = qa_chain.invoke({"query": user_query})

# Step 7: Output
print("\nResult:\n", response["result"])
print("\nSource Documents:\n", response["source_documents"])
