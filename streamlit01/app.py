import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

st.title('나만의 chatGPT')

# 처음 1번 실행. 딕셔너리에 messages를 key로 하고 value를 리스트로 한다.
if 'messages' not in st.session_state:
    st.session_state['messages'] = [] 

with st.sidebar:
    clear_btn = st.button('대화 다시')

    # if clear_btn:
    #     st.write('버튼이 눌러졌습니다.')

# 저장된 대화 출력
def print_messages():
    for chat_message in st.session_state['messages']:
        with st.chat_message(chat_message.role):
            st.write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

def create_chain():

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '당신은 친절한 AI 어시스턴트입니다.'), 
            ('user', '#Questin:\n{question}'),
        ]
    )

    llm = ChatOpenAI(
        api_key=key, 
        model_name='gpt-4o-mini',
        temperature=0			
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain


if clear_btn:
    st.session_state['messages'] = []

print_messages()

user_input = st.chat_input('궁금한 내용을 물어보세요')

if user_input:
    with st.chat_message('user'):
        st.write(user_input)
    
    chain = create_chain()
    response = chain.stream({'question': user_input})

    with st.chat_message('assistant'):
        container = st.empty()

        answer = ''

        for token in response:
            answer += token
            container.markdown(answer)

    add_message('user', user_input)
    add_message('assistant', answer)
