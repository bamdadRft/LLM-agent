import autogen
from autogen import ConversableAgent, register_function
from exa_py import Exa
import exa_py
from typing import Annotated
'''
    A simple starting point with autogen
'''


config_list = [
    {
        "model": "ollama/llama2",  
        "base_url": "http://localhost:4000",
        "api_key" : "NULL",
    }
]

llm_config = {
    "config_list" : config_list
}


exa = Exa(api_key="abf9c867-10ad-4cbd-be1c-324c1941bd1f")


def wenSearch(query : Annotated[str, "its the thing that shoulb be searched"]) -> exa_py.api.SearchResponse:
    result = exa.search(
        query,
        type="neural",
        use_autoprompt=True
        )
    return result

summarizer = ConversableAgent(
    name="summarizer",
    system_message="You are an assistant in finding best articles related to define topics and then summarize them deeply",
    llm_config=llm_config,
)

userProxy = ConversableAgent(
    name="user-proxy",
    system_message="you are a proxy user who secures the interaction and reply to different tool requests",
    llm_config=llm_config,
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
)

# mathmatition.register_for_llm(name="calculator", description="Its simple calculator with four operations that are summation, subtraction, multiplication and division")(calculator)
# userProxy.register_for_execution(name="calculator")(wenSearch)

register_function(
    wenSearch,
    caller=summarizer,
    executor=userProxy,
    name="web-search",
    description="its a web search tool which can search about any query and find related information it can be either arxiv or anything else "
)

task = "Find the related articles to the prompt engineering"
chat_res = userProxy.initiate_chat(summarizer, message=task)
