```python
import streamlit as st

from dotenv import load_dotenv
import os

from langchain_core.messages import ChatMessage

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

st.title('챗봇 만들기 - RAG')

# 처음 1번 실행
if 'messages' not in st.session_state:
    st.session_state['messages'] = []  


# 저장된 대화 내용 출력
def print_messages():
    for chat_message in st.session_state['messages']:
        with st.chat_message(chat_message.role):
            st.write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

# 저장된 대화 내용 출력
print_messages()

# 사용자 질문
user_input = st.chat_input('궁금한 내용을 물어보세요')

if user_input:

    with st.chat_message('user'):
        st.write(user_input)
    
    add_message('user', user_input)


```
