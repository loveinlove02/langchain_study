```python
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

if not os.path.exists('.cache'):
    os.mkdir('.cache')

if not os.path.exists('.cache/files'):
    os.mkdir('.cache/files')

if not os.path.exists('.cache/embeddings'):
    os.mkdir('.cache/embeddings')

st.title('챗봇 만들기')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []


with st.sidebar:
    upload_file = st.file_uploader('파일 업로드', type=['pdf'])
    select_prompt = 'prompts/pdf-rag.yaml'


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state['messages'].append(
        ChatMessage(role=role, content=message)
    )


# 저장된 메시지를 출력
def print_message():
    for chat_message in st.session_state['messages']:
        with st.chat_message(chat_message.role):
            st.write(chat_message.content) 


def embed_file(file):
    pass


if upload_file:
    retriever = embed_file(upload_file)

# 메시지 출력
print_message()

user_input = st.chat_input('궁금한 내용을 물어보세요')

if user_input:
    with st.chat_message('user'):
        st.write(user_input) 

    add_message('user', user_input)
```
