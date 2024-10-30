
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from dotenv import load_dotenv
import os
import glob

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

st.title('나만의 chatGPT')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []      

with st.sidebar:
    clear_btn = st.button('대화 다시')   

    prompt_files = glob.glob('prompts/*.yaml')  # prompts 폴더에서 파일을 가져온다 
    selected_prompt = st.selectbox('프롬프트를 선택해 주세요.', prompt_files, index=0)

def print_messages():
    for chat_message in st.session_state['messages']:
        with st.chat_message(chat_message.role):          # 이모티콘(assistant, 사용자)
            st.write(chat_message.content)                # 내용(시스템 답변, 사용자 입력)

def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

def create_chain(prompt_file_path):
    prompt = load_prompt(prompt_file_path, encoding='utf-8')

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
    
    chain = create_chain(selected_prompt)
    response = chain.stream({'question': user_input})

    with st.chat_message('assistant'):
        container = st.empty()                           

        answer = ''

        for token in response:
            answer += token
            container.markdown(answer)


    add_message('user', user_input)
    add_message('assistant', answer)
