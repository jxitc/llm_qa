from flask import Flask, request
import time
from llm_qa import LlmQA
import json


app = Flask(__name__)
qa = LlmQA([])  # empty allowlist

@app.route('/prepare_doc', methods=['POST'])
def init_doc():
    data = request.get_json()
    query = data['query']
    qa.prepare_doc(init_query=query)
    return json.dumps({'sucess': True})

@app.route('/query', methods=['POST'])
def post_data():
    start_time = time.time()
    data = request.get_json()
    query = data['query']
    answer, debug_docs = qa.answer(query)
    elapsed_time = time.time() - start_time
    debug = {'similar_docs': debug_docs, 'elapsed_time_sec': elapsed_time}
    return json.dumps({'sucess': True, 'answer': answer, 'debug': debug}, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)

