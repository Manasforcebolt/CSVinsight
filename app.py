import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
# from app import *
from model.llm import llm



DB_FAISS_PATH = 'vectorstore/db_faiss'

st.title("CSVInsight: Unlocking Data Brilliance")
st.markdown("<h3 style='text-align:center; color:red'>Built by <a href='https://github.com/Manasforcebolt'>Manas Chauhan with ❤️ </a></h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name



loader = CSVLoader(file_path=tmp_file_path, encoding='utf-8', csv_args={
    'delimiter':','
})
data = loader.load()
embeddings = HuggingFaceEmbeddings()

db= FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)


chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever()
)





def conversational_chat(query):
    result = chain(
        {"question":query,
         "chat_history":st.session_state['history']}
    )
    st.session_state['history'].append((query,result['answer']))
    return result['answer']


if 'history' not in st.session_state:
    st.session_state['history']=[]

if 'generated' not in st.session_state:
    st.session_state['generated'] = ['Hello! Ask me anything about '+ uploaded_file.name + " 🤗 "]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! 👋"]


response_container = st.container()
container = st.container()


with container:
    with st.form(key='my_form',clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:)", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input :
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="personas")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

