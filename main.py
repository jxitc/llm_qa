from flask import Flask, request
from llm_qa import LlmQA
import json


app = Flask(__name__)
qa = LlmQA([])  # empty allowlist

@app.route('/prepare_doc', methods=['POST'])
def init_doc():
    qa.prepare_doc(init_query=None)
    return json.dumps({'sucess': True})

@app.route('/query', methods=['POST'])
def post_data():
    data = request.get_json()
    query = data['query']
    answer = qa.answer(query)
    return json.dumps({'sucess': True, 'answer': answer}, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)

