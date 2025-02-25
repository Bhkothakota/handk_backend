from fastapi import FastAPI, HTTPException
from azure.cosmos import CosmosClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core import output_parsers
from langchain_community.cache import AzureCosmosDBNoSqlSemanticCache
from models import SearchRequest, OutputFormat, Text, Document
from dotenv import load_dotenv
import os
import json
import requests
from prompt import template

load_dotenv()
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_version = os.getenv("AZURE_OPENAI_VERSION")
azure_search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
cosmosdb_connection_string = os.getenv("COSMOSDB_CONNECTION_STRING")
index = os.getenv("INDEX_NAME")
azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
semantic_configuration_name = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION")
cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
cosmos_key = os.getenv("COSMOS_KEY")

app = FastAPI()

# vector_embedding_policy = {
#     "vectorEmbeddings": [
#         {
#             "path": "/contentVector",
#             "dataType": "float32",
#             "dimensions": 1536,
#             "distanceFunction": "cosine"
#         }
#     ]
# }

# indexing_policy = {
#     "automatic": True,
#     "includedPaths": [{"path": "/*"}],
#     "excludedPaths": [{"path": "/contentVector/*"}],
#     "vectorIndexes": [{"path": "/contentVector", "type": "quantizedFlat"}]
# }

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_key=openai_api_key,
    api_version=openai_version,
    azure_endpoint=openai_endpoint,
    temperature=0
)

query_enhancer = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_key=openai_api_key,
    api_version=openai_version,
    azure_endpoint=openai_endpoint,
    temperature=0
)

search_client = SearchClient(
    endpoint=azure_search_endpoint,
    index_name=index,
    credential=AzureKeyCredential(azure_search_api_key)
)

# embedding = OpenAIEmbeddings(
#     deployment="text-embedding-ada-002",
#     openai_api_key=openai_api_key
# )

# cosmos_client = CosmosClient(
#     url=cosmos_endpoint,
#     credential=cosmos_key
# )

# llm.cache = AzureCosmosDBNoSqlSemanticCache(
#     embedding=embedding,
#     cosmos_client=cosmos_client,
#     vector_embedding_policy=vector_embedding_policy,
#     indexing_policy=indexing_policy,
#     cosmos_container_properties={"id": "", "partition_key": {"paths": ["/query"]}},
#     cosmos_database_properties={"id": ""}
# )

structured_llm = llm.with_structured_output(OutputFormat)
LLM_STRING = f"{llm.deployment_name}"

history = []

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Chunk the text into smaller chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def embed_documents(query:str):
    url = f"{openai_endpoint}/openai/deployments/text-embedding-ada-002/embeddings?api-version={openai_version}"
    headers = {
            "Content-Type": "application/json",
            "api-key": openai_api_key
        }
    data = {"input": query}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data)).json()    
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def add_to_history(qapair):
    # Truncate history if it exceeds 5
    if len(history) == 5:
        history.pop(0)
    history.append(qapair)

def format_qapair(question, answer):
    return [{"role": "user", "content": question},
            {"role": "Sophie", "content": answer}]

def lookup(query):
    return llm.cache.lookup(query, LLM_STRING)

def cache(query, response):
    llm.cache.update(query, LLM_STRING, response)

def enhance_query(query, history):
    template = """You are a component of an RAG (Retrieval Augmented Generation) chain that takes in a user query and conversation history to output a self-contained query for document retrieval. 
    Your task is to modify the user query to be completely independent of relavant conversation context. 
    Use only the previous questions from the conversation history to resolve ambiguities in the new query. 
    Assume that questions towards the end of the conversation history list are more related to the current query as opposed to ones at the first. 
    If the query is in a language different from English, convert it to English. 
    Respond with a JSON parseable string. If the user query is already self-contained, return it without modifications.

    HISTORY: "[{{"role":"user", "content":"What is Python used for?"}}, {{"role":"chatbot", "content":"Python is a versatile programming language commonly used for web development, data analysis, AI, and automation."}}]"
    QUERY: "Is it used in machine learning?"

    {{"Modified":"Is Python used in machine learning?"}}

    HISTORY: "[{{"role":"user", "content":"What is Python used for?"}}, {{"role":"chatbot", "content":"Python is a versatile programming language commonly used for web development, data analysis, AI, and automation."}}]"
    QUERY: "How is python used in AI?"

    {{"Modified":"How is python used in AI?"}}

    USER QUERY: {query}
    HISTORY: {history}"""

    prompt = ChatPromptTemplate.from_template(template)
    parser = output_parsers.JsonOutputParser()
    chain = prompt | query_enhancer | parser
    return chain.invoke({"query":query, "history":history})["Modified"]

def azure_search(query):
    vector_query = VectorizedQuery(vector=embed_documents([query]), k_nearest_neighbors=3, fields="vector")
    search_options = {
            "search_text":query,
            "query_type":"semantic",
            "semantic_configuration_name":semantic_configuration_name,
            "top":5,
            "query_answer":"extractive",
            "include_total_count":False,
            "select":["url", "title", "content"],
            "vector_queries":[vector_query]
        }
    search_results = search_client.search(**search_options)
    docs = list()
    # docs.append({"answer":search_results.get_answers()})
    for result in search_results:
        doc = {
            "title" : result.get("title"),
            "url" : result.get("url"),
            "content" : result.get("content")
        }
        docs.append(doc)
    return docs

def get_completion(docs, query, history):
    # template = """
    # - You are Sophie, a helpful female AI assistant.
    # - You assist users such as the sales executive, business developement executive, pre-sales executive or any other employee by answering their queries based on provided context and conversation history.
    # - Provided conversation history contains the last few interactions between you and the user. The ones at the end of the list are more recent than ones at the top.
    # - Resolve ambiguities in current user query if any by referring to the user's previous questions.
    # - Answer the query based solely on the provided context.
    # - Include inline citations using [[number]](doc_url) format whenever you reference information from the provided documents.
    # - For greetings or general conversation, respond politely but keep it brief.
    # - Ensure that all distinct sources used to answer user query are clearly listed under the "source" field with URLs included. 
    # - Ensure accuracy and avoid any hallucinations or irrelevant information.
    # - Answer in the same language as that of user query.
    # - If multiple documents contain relevant information, combine the information coherently.
    # - Add an array of 2 specific, self-contained question objects that probe deeper into the topic under the "followup" field. Make sure to suggest only those questions that can be answered by the provided context.
    # - Leave source and followup fields empty for general greetings or when answer to user's query is not provided in context.
    
    # USER CONTEXT : {context}
    # USER QUERY : {query}
    # CONVERSATION HISTORY : {history}
    # """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | structured_llm 
    return chain.invoke({"context":docs,"query":query, "history":history})

@app.get("/")
async def root():
    try:
        return {"This is an" : "AI assistant"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/uploadfile/")
async def upload_file(file: Text):
    try:
        # Read the file content
        text = str(file)
        # Chunk the text
        chunks = chunk_text(text)
        # Return the chunks as JSON
        return JSONResponse(content={"chunks": chunks})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")
    
@app.post("/index")
async def index_data_endpoint(documents: Document):
    try:
        # Prepare documents for indexing
        # docs_to_index = [doc.dict() for doc in documents]
 
        # Index documents
        result = search_client.upload_documents(documents=documents)
        response = {
            "message": "Documents indexed successfully",
            # "results": [
            #     {"document_id": r["parent_id"], "status": r["status"]} for r in result
            # ],
        }
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error indexing documents: {e}")
    
@app.post("/search")
async def search_documents(searchRequest: SearchRequest):
    try:
        query = searchRequest.search_text
        # result = lookup(query)
        # if result:
        #     return result
        enhanced_query = enhance_query(query, history)
        # print(enhanced_query)
        # result = lookup(enhanced_query)
        # if result:
        #     return result
        docs = azure_search(enhanced_query)
        completion = get_completion(docs, query, history)
        # cache(enhanced_query, completion)
        add_to_history(format_qapair(query, completion.response))
        return completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))