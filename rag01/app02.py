
import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os



load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

# 캐시 디렉토리 생성
if not os.path.exists('.cache'):
    os.mkdir('.cache')

if not os.path.exists('.cache/files'):
    os.mkdir('.cache/files')

if not os.path.exists('.cache/embeddings'):
    os.mkdir('.cache/embeddings')


st.title('나만의 chatGPT')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'chain' not in st.session_state:
    st.session_state['chain'] = None

with st.sidebar:
    clear_btn = st.button('대화 다시')    

    upload_file = st.file_uploader('파일 업로드', type=['pdf'])

    selected_prompt = 'prompts/pdf-rag.yaml'



def print_messages():                                     
    for chat_message in st.session_state['messages']:
        with st.chat_message(chat_message.role):          # 이모티콘(assistant, 사용자)
            st.write(chat_message.content)                # 내용(시스템 답변, 사용자 입력)


def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))


# 파일을 캐시해서 저장
@st.cache_resource(show_spinner='업로드한 파일을 처리 중입니다...')
def embed_file(file):      
    file_content = file.read()
    file_path = f'./.cache/files/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    
    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


def create_chain(retriever):
    # 프롬프트 생성(Create Prompt)
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

    # 언어모델(LLM) 생성
    llm = ChatOpenAI(
        api_key=key, 
        model_name='gpt-4o-mini',
        temperature=0,			    
        max_tokens=2048,			
    )

    # 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}  #  itemgetter('question')
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if upload_file:                             
    retriever = embed_file(upload_file)     # 파일이 업로드되면 retriever 
    chain = create_chain(retriever)         # retriever를 인자로 넘겨서 체인
    st.session_state['chain'] = chain       # session_state에 체인을 등록

# 버튼이 눌러지면
if clear_btn:
    st.session_state['messages'] = []

print_messages()

user_input = st.chat_input('궁금한 내용을 물어보세요')  

warning_msg = st.empty()                            # 경고 메시지 출력을 위한 빈 영역을 잡는다

if user_input:                                      # 사용자 입력이 들어있으면

    chain = st.session_state['chain']               # session_state에서 체인을 가져오기

    if chain is not None:                           # session_state에서 체인을 가져왔으면

        with st.chat_message('user'):               # 사용자 이모티콘
            st.write(user_input)                    # 사용자가 입력한 내용(질문)을 출력
            
        # RunnablePassthrough()를 사용하기 때문에 user_input만 인자로 넣는다.
        response = chain.stream(user_input)

        # response = chain.stream({'question': user_input})
        # 위의 것을 사용하고 싶으면 create_chain() 함수에서 
        # RunnablePassthrough()를 itemgetter('question')로 바꾼다.

        with st.chat_message('assistant'):          # assistant 이모티콘
            container = st.empty()                  # 빈공간을 만든다

            answer = ''

            for token in response:
                answer += token
                container.markdown(answer)

        add_message('user', user_input)             # 대화 내용을 기록한다
        add_message('assistant', answer)
    else:                                           # session_state에서 체인을 못 가져왔으면
        warning_msg.error('파일을 업로드 해주세요.')

