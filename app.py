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

def main():
    print(st.secrets)
    load_dotenv()
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # settings
        models_url = "https://platform.openai.com/docs/models"
        model = st.selectbox('Choose the model ([learn more](%s))' % models_url,
                              ('gpt-3.5-turbo-16k',
                               'gpt-3.5-turbo',
                               'text-davinci-003',
                               'gpt-4'))

        # show user input
        st.write(" ")
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(temperature=0.1, model_name=model)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)
                print(cb)
                print(response)

            st.write(response)

            # response detail
            st.write(" ")
            st.write(cb)  
            st.markdown(f"<span style='color: gray; font-size: 11px;'>Model: {model}</span>",
             unsafe_allow_html=True)


if __name__ == '__main__':
    main()
