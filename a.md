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


st.title('챗봇 만들기')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []


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

# 체인을 생성
def create_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '당신은 친절한 AI 어시스턴스입니다.'), 
            ('user', '#Question:\n{question}')
        ]
    )

    llm = ChatOpenAI(
        api_key=key, 
        model_name = 'gpt-4o-mini'
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    
    return chain

# 메시지 출력
print_message()

user_input = st.chat_input('궁금한 내용을 물어보세요')

if user_input:
    with st.chat_message('user'):
        st.write(user_input) 


    # 체인 함수를 실행 해서 chain 얻기
    chain = create_chain()
    answer = chain.invoke({'question': user_input})

    # 화면에 답변 출력
    with st.chat_message('assistant'):
        st.write(answer)

    add_message('user', user_input)
    add_message('assistant', answer)
```
