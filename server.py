from flask import Flask
from flask_cors import CORS
import json
from pubmad import get_graph
import networkx as nx

app = Flask(__name__)
CORS(app)

@app.route("/get_graph/<string:query>/<int:max_publications>")
def API_get_graph(query, max_publications):
    G = get_graph(query, max_publications=max_publications)
    cy_data = json.dumps(nx.readwrite.json_graph.cytoscape_data(G))
    return cy_data