import flask
from flask import Flask
from flask import render_template
from flask import send_file
from flask import request

import os
import json
import logging
import queue
from gpt.openai_chat import openai_qa

app = Flask(__name__)
chat_history=[]

# Ugh!: will fail if there are more than two clients accessing /listen
qa_stream = queue.Queue()

# loading chat client
@app.route('/')
def chat_client():
    return render_template('chat_client.html')

# icon botPhoto
@app.route('/icon/botPhoto')
def icon():
    icon = os.path.dirname(__file__) + '/'+ 'templates/botPhoto.png'
    logging.debug("icon:" + icon)
    return send_file(icon, mimetype='image/png')

def dummy_callback(token):
    qa_stream.put(token)
    print('callback>> \033[36m' + token + '\033[0m')

@app.route('/chat')
def chat():
    # preprocess
    query = request.args.get('text')
    callback = request.args.get('callback')
    # query to openai
    chat_history = []
    result = openai_qa(query, chat_history, dummy_callback)
    answer = result["answer"]
    # debug
    logging.debug("answer: " + answer)
    for doc in result['source_documents']:
        logging.debug(doc.page_content.split('\n', 1)[0])
        logging.debug(doc.metadata)
    # answer to client
    res = callback + '(' + json.dumps({
                "output":[  
                    {  
                        "type":"text",
                        "value":answer
                    }
                    ]
            }) + ')'
    logging.debug(res)
    return res

# Ugh!: No end is reached because None is never put into qa_stream.
@app.route('/listen')
def listen():
    def stream():
        while True:
            msg = qa_stream.get()
            if msg is None:
                break 
            yield f'data: {msg}\n\n'
    response = flask.Response(stream(), mimetype='text/event-stream')
    return response

# AirPlay uses port 5000 on Mac
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s")   
    app.debug = True
    app.run(port='5001', threaded=True)

if __name__ == '__main__':
    main()