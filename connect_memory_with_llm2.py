from dotenv import load_dotenv
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS



LOCAL_MODEL_PATH =r"D:\project\AI_MEDIBOT\google_flan_t5_small\snapshots\flan_t5_small" # Change if needed
DB_FAISS_PATH = r"vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = r"sentence-transformers/all-MiniLM-L6-v2"


# Step 1: Load environment variables
def load_env_variables():
    load_dotenv()

# Step 2: Setup LLM (Local Model)
def load_local_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    local_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.5,
        do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=local_pipeline)
    return llm

# Step 3: Setup custom prompt
def set_custom_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Step 4: Load vectorstore (FAISS)
def load_vectorstore(db_path, embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Step 5: Create Retrieval QA chain
def create_qa_chain(llm, vectorstore, prompt_template):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt(prompt_template)},
    )
    return qa_chain

# Step 6: Main function
def main():
    load_env_variables()

    # Define paths

    
    # Custom prompt template
    custom_prompt_template = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say you don't know — do not try to make up an answer.
    Only use the information provided.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk.
    """

    # Load all components
    llm = load_local_llm(LOCAL_MODEL_PATH)
    vectorstore = load_vectorstore(DB_FAISS_PATH, EMBEDDING_MODEL_NAME)
    qa_chain = create_qa_chain(llm, vectorstore, custom_prompt_template)

    # Query the model
    user_query = input("Write your query here: ")
    response = qa_chain.invoke({"query": user_query})

    # Output the result
    print("\nResult:\n", response["result"])
    print("\nSource Documents:\n", response["source_documents"])

# Step 7: Entry point
if __name__ == "__main__":
    main()
