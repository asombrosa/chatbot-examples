import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import StrOutputParser
import getpass

load_dotenv()
os.environ["AZURE_OPENAI_API_KEY"]
model = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
)
"""client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-02-01"
)"""

LANGUAGE_MODEL = "gpt-4o"

system_prompt = "You are a helpful assistant that answers generals inquiries and assist with technical issues"

str_parser = StrOutputParser()

# basic example of how to get started with the OpenAI Chat models
# The above cell assumes that your OpenAI API key is set in your environment variables.
#model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


def main():
    user_input = "I want to return a pair of shoes"

    # prompt value
    prompt_value = chat_prompt.invoke({"question": user_input})
    # print(prompt_value.to_string())

    # model response
    messages = chat_prompt.format_prompt(question=user_input).to_messages()
    response = model.invoke(messages)
    print(response)

    # string output parser
    content = str_parser.invoke(response)
    print(content)

    # LCEL makes it easy to build complex chains from basic components, and supports out of the box functionality such as streaming, parallelism, and logging.


if __name__ == "__main__":
    main()
