#import openai
import os
import tiktoken
from colorama import Fore
from dotenv import load_dotenv
from openai import AzureOpenAI


# Load the environment variables - set up the OpenAI API client
load_dotenv()
client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2023-03-15-preview"
)
#client.api_key =   os.getenv("OPENAI_API_KEY")
# ###
# Set up the model and prompt
LANGUAGE_MODEL = "gpt-4o"
PROMPT_TEST = "This is a test prompt. Say this is a test"


def get_tokens(user_input: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding("cl100k_base")

    token_integers = encoding.encode(user_input)
    tokens_usage = len(token_integers)

    tokenized_input = tokenized_input = list(
        map(
            lambda x: encoding.decode_single_token_bytes(x).decode("utf-8", errors='ignore'),
            encoding.encode(user_input),
        )
    )
    print(f"{encoding}: {tokens_usage} tokens")
    print(f"token integers: {tokens_usage}")
    print(f"token bytes: {tokenized_input}")


def start():
    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        exit()
    else:
        print("Invalid choice")


def ask():
    """Ask a question and get a response from the model."""
    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    while True:
        user_input = input("Q: ")

        # Exit
        if user_input == "x":
            start()
        else:
            completion = client.chat.completions.create(
                model="gpt-4o", # prompt=str(user_input),
                max_tokens=100,
                temperature=0,
                messages=[
                    {"role": "system", "content": "you are a funny chatbot, make me laugh"},
                    {"role": "user", "content": str(user_input)}
                ],
            )

            response = completion.choices[0].message.content
            #("\n", "").replace("\r", "")

            get_tokens(response)

            print(Fore.BLUE + f"A: {response}" + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
