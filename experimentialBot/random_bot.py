import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from colorama import Fore, init
init()  # Initialize colorama

from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15"
)
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2023-05-15"
)

os.environ["AZURE_OPENAI_API_KEY"]
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment="gpt-4",
    openai_api_version="2023-05-15"
)

system_prompt = """You are a helpful assistant that answers inquiries and assists with technical issues.
Answer the question based ONLY on the context provided below. If you don't know the answer or the information 
is not in the context, say "I don't have enough information to answer that" instead of making up an answer.

Context:
{context}
"""

str_parser = StrOutputParser()

def process_query():
    # Load documents with error handling
    try:
        loader = DirectoryLoader('kargo', use_multithreading=True, loader_cls=TextLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

    # Use appropriate chunk size for meaningful context (1000-1500 chars instead of 50)
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    print(f"Split into {len(documents)} chunks")

    # Create vector store
    db = Chroma.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})  # Get top 3 most relevant docs
    
    return retriever

def create_rag_chain(retriever):
    # Create the system and human message templates
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
    
    # Create a chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    # Create a chain that combines the retrieved documents with the question
    document_chain = create_stuff_documents_chain(
        model, 
        chat_prompt
    )
    
    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain

def print_messages(messages):
    messages = [message for message in messages if message["role"] != "system"]
    for message in messages:
        role = "Bot" if message["role"] == "assistant" else "You"
        print(Fore.BLUE + role + ": " + Fore.RESET + message["content"])
    return messages

if __name__ == "__main__":
    messages = []
    retriever = process_query()
    
    if retriever:
        rag_chain = create_rag_chain(retriever)
        
        while True:
            user_query = input(Fore.GREEN + "Enter your query (or 'x' to exit): " + Fore.RESET)
            if user_query.lower() == 'x':
                break
                
            # Execute the RAG chain
            result = rag_chain.invoke({"question": user_query})
            
            # Print the answer and the source documents for transparency
            print(Fore.CYAN + "Bot: " + Fore.RESET + result["answer"])
            print(Fore.YELLOW + "\nSources:" + Fore.RESET)
            for i, doc in enumerate(result.get("context", [])):
                print(f"{i+1}. {doc.page_content[:100]}...")
            
            # Add to chat history
            messages.append({"role": "user", "content": user_query})
            messages.append({"role": "assistant", "content": result["answer"]})

