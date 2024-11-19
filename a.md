```python
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

# 캐시 디렉토리를 생성
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
    selected_prompt = 'prompts/pdf-rag.yaml'


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


@st.cache_resource(show_spinner='업로드한 파일을 처리 중입니다...')
def embed_file(file):      
    file_content = file.read()
    file_path = f'./.cache/files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)

    # 1. 문서로드
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # 2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50
    )

    split_documents = text_splitter.split_documents(docs)

    # 3. 임베딩기 생성
    embeddings = OpenAIEmbeddings()

    # 4. 벡터 데이터베이스(faiss) 생성 및 저장
    vectorstore = FAISS.from_documents(
        documents=split_documents, embedding=embeddings
    )

    # 5. 검색기(retriever) 생성
    retriever = vectorstore.as_retriever()
    
    return retriever


def create_chain(retriever):
    
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Question: 
    {question} 
    #Context: 
    {context} 

    #Answer:"""
    )

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(
        api_key=key, 
        model_name='gpt-4o-mini',
        temperature=0,			    
        max_tokens=2048,			
    )

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}  #  itemgetter('question')
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# 파일을 업로드 했으면
if upload_file:                             
    retriever = embed_file(upload_file)     # 파일이 업로드되면 retriever 
    chain = create_chain(retriever)         # retriever를 인자로 넘겨서 체인
    st.session_state['chain'] = chain       # session_state에 체인을 등록


# 저장된 대화를 화면에 출력
print_message()

# 사용자 질문입력
user_input = st.chat_input('궁금한 내용을 물어보세요')

# 빈 공간
warring_msg = st.empty()

if user_input:      # 입력했으면
    # session_state에 등록시킨 체인을 가져온다
    chain = st.session_state['chain']

    # 체인을 가져왔으면
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
        add_message('user', answer)
    else:
        warring_msg.error('실패')    


    
```
