#Some of the packaches in requirements.txt are not obvious
#libmagic, python-magic and python-magic-bin are required in order
#to make UnstructuredURLLoader and FAISS work as expected

import streamlit as st
import os
import pickle
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

file_path = "faiss_store_openai.pkl"

#LLM and key loading function
def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm


#Page title and header
st.set_page_config(page_title="Ask from URLs")
st.header("Ask from URLs")


#Input OpenAI API Key
def get_openai_api_key():
    input_text = st.text_input(
        label="OpenAI API Key ",  
        placeholder="Ex: sk-2twmA8tfCb8un4...", 
        key="openai_api_key_input", 
        type="password")
    return input_text

openai_api_key = get_openai_api_key()

urls = []
for i in range(3):
    url = st.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.button("Click to load the URLs")

if process_url_clicked:
    if not openai_api_key:
        st.warning('Please insert OpenAI API Key. \
            Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', 
            icon="⚠️")
        st.stop()
    
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=500
    )

    docs = text_splitter.split_documents(data)
    
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)



# Input
st.markdown("## Enter your question")

def get_question():
    question_text = st.text_area(
        label="Question Text", 
        label_visibility='collapsed', 
        placeholder="Enter your question here...", 
        key="question_input"
        )
    return question_text

question_input = get_question()

if len(question_input.split(" ")) > 700:
    st.write("Please enter a shorter question. The maximum length is 700 words.")
    st.stop()

if question_input:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            llm = load_LLM(openai_api_key=openai_api_key)

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, 
                retriever=vectorstore.as_retriever()
                )   
    
            result = chain({"question": question_input}, return_only_outputs=True)
    
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])
    
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)