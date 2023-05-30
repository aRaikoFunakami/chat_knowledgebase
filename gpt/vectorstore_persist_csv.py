import os
import sys
import json

import openai
import logging

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

# debug config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")

# load config
args = sys.argv
for i in args:
    logging.debug(i)
    
config_file = os.path.dirname(__file__) + "/config.json" if len(args) <=1 else args[1]

logging.debug(config_file)
with open(config_file, 'r') as file:
    config = json.load(file)

for key, value in config.items():
    logging.debug(key +":" + value)

# config
openai_api_key = config['openai_api_key']
file_path = os.path.dirname(__file__) + '/'+ config['file']
persist_dir =os.path.dirname(__file__) + '/'+ config['persist_directory']

# load 
loader = CSVLoader(file_path=file_path,  source_column=config.get('source_column'))
data = loader.load()

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
)

docs = text_splitter.split_documents(data)
logging.debug(docs[0])

# Preprocessing for using Open　AI
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

# Stores information about the split text in a vector store
vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
vectorstore.persist()