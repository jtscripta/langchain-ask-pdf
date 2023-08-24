from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
import os

st.set_page_config(page_title="Ask your PDF")
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
langchain.llm_cache = InMemoryCache()

models_url = "https://platform.openai.com/docs/models"
temp_url = "https://www.linkedin.com/pulse/temperature-check-guide-best-chatgpt-feature-youre-using-berkowitz"

@st.cache_data
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def split_text(text):
    text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
    return text_splitter.split_text(text)

@st.cache_data
def create_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base


def user_input_section(knowledge_base, model, temp):
    # user_question = st.text_input("Ask a question about your PDF:")
    user_question = st.chat_input("Ask a question about your PDF:")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner(""):
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(temperature=float(temp), model_name=model)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
                print(response)
            with st.chat_message("assistant"):
                st.write(response)
                st.write(" ")
                st.write(cb)
                st.markdown(
                    f"<span style='color: gray; font-size: 11px;'>Model: {model}</span>",
                    unsafe_allow_html=True,
                )


def main():
    load_dotenv()
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        text = extract_text_from_pdf(pdf)

        # split into chunks
        chunks = split_text(text)

        print("Embedding to run: ...")

        # create embeddings
        knowledge_base = create_embeddings(chunks)

        # settings
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox(
                "Choose the model ([learn more](%s))" % models_url,
                ("gpt-3.5-turbo-16k", "gpt-3.5-turbo", "text-davinci-003", "gpt-4"),
            )
        with col2:
            temp = st.text_input(
                "Temperature (0.1 - 1.0 [learn more](%s))" % temp_url, 0.1
            )

        # show user input
        st.divider()
        user_input_section(knowledge_base, model, temp)


if __name__ == "__main__":
    main()
