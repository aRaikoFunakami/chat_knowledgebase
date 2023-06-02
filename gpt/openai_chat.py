import os
import sys
import logging
import json

import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.manager import CallbackManager, BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# load config
def load_config():
    args = sys.argv
    config_file = os.path.dirname(__file__) + "/config.json" if len(args) <=1 else args[1]
    
    logging.info(config_file)
    with open(config_file, 'r') as file:
        config = json.load(file)

    return {
            "openai_api_key": config['openai_api_key'],
            "persist_dir": os.path.dirname(__file__) + '/'+ config['persist_directory']
        }
    
    
# Streaming対応
# ref: https://python.langchain.com/en/latest/modules/callbacks/getting_started.html
# ref: https://ict-worker.com/ai/langchain-stream.html
from typing import Any, Dict, List, Optional, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult

class MyCustomCallbackHandler(BaseCallbackHandler):
    def __init__(self, callback):
        self.callback = callback
    """Custom CallbackHandler."""
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.callback is not None:
            self.callback(token) 

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""


# Preprocessing for using OpenAI
def get_chain(config, callback_streaming=None):
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # callback
    callback_manager = CallbackManager([MyCustomCallbackHandler(callback_streaming)])
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", callback_manager=callback_manager, streaming=True)
    embeddings = OpenAIEmbeddings()
    # Loading the persisted database from disk
    vectordb = Chroma(persist_directory=config["persist_dir"], embedding_function=embeddings)
    # IF for asking OpenAI
    return ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), return_source_documents=True)


def openai_qa(query, history=[], callback_streaming=None):
    csv_qa = get_chain(load_config(), callback_streaming)
    chat_history = history
    return csv_qa({"question": query, "chat_history": chat_history})


def dummy_callback(token):
    print('callback>> \033[36m' + token + '\033[0m')
    
def main():
    # Question
    query = 'ラズパイでスクリーンショットを撮りたい'
    result = openai_qa(query, [], dummy_callback)
    print(result["answer"])
    # Review the original set of texts related to the above responses.
    for doc in result['source_documents']:
        print(doc.page_content.split('\n', 1)[0])
        print(doc.metadata)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")    
    main()