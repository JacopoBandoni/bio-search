from flask import Flask
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route("/get_graph/<string:query>")
def get_graph(query):
    return json.dumps({"response": query})