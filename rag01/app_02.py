
from dotenv import load_dotenv
import os

load_dotenv(verbose=True)
key = os.getenv('OPENAI_API_KEY')

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


# 1. 문서 로드(Load Documents)
loader = PyMuPDFLoader("./SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()


# 2. 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)


# 3. 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()


# 4. DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)


# 5. 검색기(Retriever) 생성
# 문서에 포함되어있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()


# 6. 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
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

# 7. 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(
    api_key=key, 
    model_name='gpt-4o-mini',
    temperature=0.1,
    max_tokens=2048,
)


# 8. 체인(Chain) 생성
chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | prompt
    | llm 
    | StrOutputParser()
)


question = "삼성전자가 자체 개발한 AI 의 이름은?"
response = chain.invoke(question)

print(response)



















