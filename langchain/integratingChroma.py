import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import StrOutputParser

from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()




system_prompt = "You are a helpful assistant that answers generals inquiries and assist with technical issues"

str_parser = StrOutputParser()

# basic example of how to get started with the OpenAI Chat models
# The above cell assumes that your OpenAI API key is set in your environment variables.
os.environ["AZURE_OPENAI_API_KEY"]
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# load the document and split it into chunks
loader = TextLoader("./docs/faq.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print(docs)


def main():
    user_input = "Do you ship to Europe?"

    # LCEL makes it easy to build complex chains from basic components, and supports out of the box functionality such as streaming, parallelism, and logging.
    chain = chat_prompt | model | str_parser
    # message = chain.invoke({"question": user_input})
    # print(message)


if __name__ == "__main__":
    main()
