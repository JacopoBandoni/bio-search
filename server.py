from flask import Flask
from flask_cors import CORS
import json
from pubmad import get_graph
import networkx as nx

app = Flask(__name__)
CORS(app)

@app.route("/get_graph/<string:query>")
def API_get_graph(query):
    G = get_graph(query)
    cy_data = json.dumps(nx.readwrite.json_graph.cytoscape_data(G))
    return cy_data