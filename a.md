```python
import streamlit as st
from langchain_core.messages import ChatMessage

st.title('챗봇 만들기')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    st.write('처음 한 번은 만들어 짐')

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

# 메시지 출력
print_message()

user_input = st.chat_input('궁금한 내용을 물어보세요')

if user_input:
    with st.chat_message('user'):
        st.write(user_input) 

    add_message('user', user_input)

```
