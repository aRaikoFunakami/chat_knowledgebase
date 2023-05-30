import os
import sys
import logging
import json

import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

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

# Preprocessing for using OpenAI
def get_chain(config):
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()
    # Loading the persisted database from disk
    vectordb = Chroma(persist_directory=config["persist_dir"], embedding_function=embeddings)
    # IF for asking OpenAI
    return ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), return_source_documents=True)

def openai_qa(query, history=[]):
    csv_qa = get_chain(load_config())
    chat_history = history
    return csv_qa({"question": query, "chat_history": chat_history})

def main():
    # Question
    query = 'ラズパイでスクリーンショットを撮りたい'
    result = openai_qa(query)
    print(result["answer"])
    # Review the original set of texts related to the above responses.
    for doc in result['source_documents']:
        print(doc.page_content.split('\n', 1)[0])
        print(doc.metadata)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")    
    main()