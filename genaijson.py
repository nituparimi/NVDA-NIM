import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
 # from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

from langchain_community.document_loaders import JSONLoader

from pathlib import Path


# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Directory containing all JSON files
json_dir = Path("./us_census2")
all_docs = []

# Loop through all JSON files in the directory
for json_file in json_dir.glob("*.json"):
    loader = JSONLoader(
        file_path=json_file,
        jq_schema='.updates[] | .plugin.description + " " + .plugin.synopsis + " Host: " + .asset.hostname',
        text_content=True
    )
    docs = loader.load()
    all_docs.extend(docs)

print("Total loaded docs:", len(all_docs))
if all_docs:
    print("Sample content:", all_docs[0].page_content[:200])


from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

emb = NVIDIAEmbeddings()
print("Available models:", emb.available_models)



def vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize embeddings
        #st.session_state.embeddings=NVIDIAEmbeddings() #defaults to model nvidia-embed-qa-4
        st.session_state.embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2")

        # Use the docs we already loaded at the top
        st.session_state.docs = docs

        # Split docs into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=50
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:30]
        )

        print("Number of final docs:", len(st.session_state.final_documents))
        print("Sample chunk:", st.session_state.final_documents[0].page_content[:200])

        # Build FAISS vector store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")


prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


prompt1=st.text_input("Enter Your Question From Doduments")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
