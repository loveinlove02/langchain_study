```python
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote.prompts import load_prompt

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

if 'chain' not in st.session_state:
    st.session_state['chain'] = None

with st.sidebar:
    upload_file = st.file_uploader('파일 업로드', type=['pdf'])
    

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

@st.cache_resource(show_spinner='업로한 파일을 처리 중입니다...')
def embed_file(file):
    file_content = file.read()
    file_path = f'./.cache/files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)


    # 1. 문서를 로더를 사용해서 로드
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 2. 문서를 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50
    )
    split_documents = text_splitter.split_documents(docs)

    # 3. 임베딩을 생성
    embeddings = OpenAIEmbeddings()

    # 4. 벡터 DB 생성하고 문서를 저장
    vectorstore = FAISS.from_documents(
        documents=split_documents, embedding=embeddings
    )


    # 5. 질문과 비슷한 내용의 문단을 검색할 수 있는 검색기를 만든다
    retriever = vectorstore.as_retriever()

    return retriever
    

def create_chain(retriever):
    prompt = load_prompt('prompts/pdf-rag.yaml', encoding='utf-8')
    
    llm = ChatOpenAI(
        api_key=key, 
        model_name='gpt-4o-mini'
    )

    output_parser = StrOutputParser()

    chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return chain

if upload_file:
    retriever = embed_file(upload_file)
    chain = create_chain(retriever)
    st.session_state['chain'] = chain


# 메시지 출력
print_message()

user_input = st.chat_input('궁금한 내용을 물어보세요')

warning_msg = st.empty()

if user_input:
    chain = st.session_state['chain']

    if chain is not None:

        with st.chat_message('user'):
            st.write(user_input) 

        response = chain.stream(user_input)

        with st.chat_message('assistant'):
            container = st.empty()
            answer = ''

            for token in response:
                answer = answer + token
                container.markdown(answer)

        add_message('user', user_input)
        add_message('assistant', answer)
    else:
        warning_msg.error('오류')        
    
```
