# Sentence Embedding
* Loading data from a cvs file
* Storing the date to vectorstore
* Saving the vectorestore to a file
```terminal
$ python3 ./gpt/vectorstore_persist_csv.py
$ ls gpt/db
chroma-collections.parquet	chroma-embeddings.parquet	index
```

# Chat about knowledgebase
* Launching a chat app
* Browsing http://127.0.0.1:5000 
```terminal
$ python3 app.py  
* Serving Flask app 'app'
* Debug mode: on
INFO:werkzeug - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
* Running on http://127.0.0.1:5000
```

# Library
- Chat UI : [chatux](https://github.com/riversun/chatux)