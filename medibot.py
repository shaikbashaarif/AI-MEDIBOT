import streamlit as st
from dotenv import load_dotenv
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Paths
LOCAL_MODEL_PATH = r"D:\project\AI_MEDIBOT\models__google__flan_t5_small\snapshots\flan_t5_small"
DB_FAISS_PATH = r"D:\project\AI_MEDIBOT\vectorstore\db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load environment variables
load_dotenv()

# Initialize session state
def init_session():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

# Load local HuggingFace model
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

# Set custom prompt
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load vectorstore
@st.cache_resource
def load_vectorstore(db_path, embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Create a custom Retrieval QA Chain that only returns the answer (without source docs)
def create_qa_chain(llm, vectorstore, prompt_template):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,  # Don't return source documents in the result
        chain_type_kwargs={"prompt": set_custom_prompt(prompt_template)},
    )

# Main Streamlit app
def main():
    st.title("🩺 Ask Medibot!")

    init_session()

    for msg in st.session_state['messages']:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Update the prompt template with specific instructions
    prompt_template = """
    You are a medical assistant. Use only the context provided below to answer the user's question. If you don't know the answer, say 'Sorry, I don't have enough information to answer that.' Do not invent an answer or give generic advice. Only answer based on the context.

    Context: {context}
    Question: {question}

    Provide the answer directly without small talk. Be concise and medically accurate.
    """

    prompt = st.chat_input("Ask your medical question:")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state['messages'].append({"role": "user", "content": prompt})

        try:
            llm = load_local_llm(LOCAL_MODEL_PATH)
            vectorstore = load_vectorstore(DB_FAISS_PATH, EMBEDDING_MODEL_NAME)
            qa_chain = create_qa_chain(llm, vectorstore, prompt_template)

            # Run the QA chain and get the answer
            answer = qa_chain.run(prompt)

            # Display the answer
            st.chat_message("assistant").markdown(answer)

            # Append the result to the session state for chat history
            st.session_state['messages'].append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"❌ Error: {repr(e)}")

if __name__ == "__main__":
    main()




# import streamlit as st
# from dotenv import load_dotenv
# import os

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS

# # Paths
# LOCAL_MODEL_PATH = r"D:\project\AI_MEDIBOT\models__google__flan_t5_small\snapshots\flan_t5_small"
# DB_FAISS_PATH = r"D:\project\AI_MEDIBOT\vectorstore\db_faiss"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# # Load environment variables
# load_dotenv()

# # Initialize session state
# def init_session():
#     if 'messages' not in st.session_state:
#         st.session_state['messages'] = []

# # Load local HuggingFace model
# def load_local_llm(model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#     local_pipeline = pipeline(
#         "text2text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_length=512,
#         temperature=0.5,
#         do_sample=True
#     )
#     llm = HuggingFacePipeline(pipeline=local_pipeline)
#     return llm

# # Set custom prompt
# def set_custom_prompt(template):
#     return PromptTemplate(template=template, input_variables=["context", "question"])

# # Load vectorstore
# @st.cache_resource
# def load_vectorstore(db_path, embedding_model_name):
#     embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
#     db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
#     return db

# # Create Retrieval QA Chain
# def create_qa_chain(llm, vectorstore, prompt_template):
#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
#         return_source_documents=True,  # We want to get source documents as well
#         chain_type_kwargs={"prompt": set_custom_prompt(prompt_template)},
#     )

# # Main Streamlit app
# def main():
#     st.title("🩺 Ask Medibot!")

#     init_session()

#     for msg in st.session_state['messages']:
#         st.chat_message(msg["role"]).markdown(msg["content"])

#     prompt_template = """
#     Use the pieces of information provided in the context to answer user's question.
#     If you don't know the answer, just say you don't know — do not try to make up an answer.
#     Only use the information provided.

#     Context: {context}
#     Question: {question}

#     Start the answer directly. No small talk.
#     """

#     prompt = st.chat_input("Ask your medical question:")

#     if prompt:
#         st.chat_message("user").markdown(prompt)
#         st.session_state['messages'].append({"role": "user", "content": prompt})

#         try:
#             llm = load_local_llm(LOCAL_MODEL_PATH)
#             vectorstore = load_vectorstore(DB_FAISS_PATH, EMBEDDING_MODEL_NAME)
#             qa_chain = create_qa_chain(llm, vectorstore, prompt_template)

#             # Run the QA chain and get the results as a dictionary with 'result' and 'source_documents'
#             result = qa_chain.run(prompt)
            
#             # Extract the answer and source documents from the result
#             answer = result.get('result', 'No answer found.')
#             source_docs = result.get('source_documents', [])
            
#             # Display the answer
#             st.chat_message("assistant").markdown(answer)

#             # Optionally, display the source documents
#             if source_docs:
#                 sources = "\n".join([f"- {doc['page_content']}" for doc in source_docs])
#                 st.chat_message("sources").markdown(f"Source documents:\n{sources}")
#             else:
#                 st.chat_message("sources").markdown("No source documents available.")

#             # Append the result to the session state for chat history
#             st.session_state['messages'].append({"role": "assistant", "content": answer})
#             if source_docs:
#                 st.session_state['messages'].append({"role": "assistant", "content": f"Source documents: {sources}"})
#             else:
#                 st.session_state['messages'].append({"role": "assistant", "content": "No source documents available."})

#         except Exception as e:
#             st.error(f"❌ Error: {repr(e)}")

# if __name__ == "__main__":
#     main()
