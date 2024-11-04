import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import StrOutputParser

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings


from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()
client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
  api_version="2023-05-15"
)
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    openai_api_version="2023-05-15" # If not provided, will read env variable AZURE_OPENAI_API_VERSION
)

#embeddings = generate_embeddings()

os.environ["AZURE_OPENAI_API_KEY"]
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
)

system_prompt = "You are a helpful assistant that answers generals inquiries and assist with technical issues"

str_parser = StrOutputParser()

# basic example of how to get started with the OpenAI Chat models
# The above cell assumes that your OpenAI API key is set in your environment variables.
#model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


def process_query():
    #user_query = "Do you ship to Europe?"

    # LCEL makes it easy to build complex chains from basic components, and supports out of the box functionality such as streaming, parallelism, and logging.
    chain = chat_prompt | model | str_parser
    # message = chain.invoke({"question": user_input})
    # print(message)
    # load the document and split it into chunks

    loader = DirectoryLoader('golang-30', use_multithreading=True, loader_cls=TextLoader)

    #loader = TextLoader("./docs/faq.txt")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    # load documents to the vector store
    # load it into Chroma
    doc_texts = [doc.page_content for doc in documents]
    db = Chroma.from_documents(documents, embeddings)
    return db

def search_query(db, user_input):
    # query it
    docs = db.similarity_search(user_input)

    # Print results
    if docs:
        print("Top result:")
        print(docs[0].page_content)
    else:
        print("No results found.")

def print_messages(messages):
    messages = [message for message in messages if message["role"] != "system"]
    for message in messages:
        role = "Bot" if message["role"] == "assistant" else "You"
        print(Fore.BLUE + role + ": " + message["role"])
    return messages


def generate_chat_completion(user_input=""):
    messages.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    print(completion)


if __name__ == "__main__":
    db = process_query()
    while True:
        user_query = input("Enter your query (or 'x' to exit): ")
        if user_query.lower() == 'x':
            break
        search_query(db, user_query)
        
