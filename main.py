from flask import Flask, request

app = Flask(__name__)

@app.route('/post', methods=['POST'])
def post_data():
    data = request.get_json()
    print(data)
    return 'Data received'

if __name__ == '__main__':
    app.run(debug=True)

