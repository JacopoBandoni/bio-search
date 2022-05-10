from flask import Flask, request
from flask_socketio import send, emit, SocketIO
from flask_cors import CORS
import json
from pubmad import get_graph, get_communities
import networkx as nx

from pubmad.utils import display_graph

app = Flask(__name__)
CORS(app)

@app.route("/get_graph", methods=['POST'])
def API_get_graph():
    data = request.get_json()
    query = data['query']
    max_publications = int(data['max_publications'])
    start_year = data['start_year']
    end_year = data['end_year']
    use_biobert = data['use_biobert']
    use_biobert = bool(int(use_biobert))
    # convert json to dict
    nodes = data['data']['elements']['nodes']
    edges = data['data']['elements']['edges']

    if len(nodes) > 0:
        # create graph from nodes and edges from cytoscape
        elements = {'nodes': nodes, 'edges': edges}
        cy_data = {
            'elements': elements,
            'data': [],
        }
        G = nx.cytoscape_graph(cy_data)
        G = get_graph(query, max_publications, start_year, end_year, use_biobert, clear_cache=False, G=G)
    else:
        G = get_graph(query, max_publications=max_publications, start_year=start_year, end_year=end_year, use_biobert=use_biobert,
            clear_cache=False)#, callback_fn=lambda partial_G: emit('update', {'cy_data': json.dumps(nx.readwrite.json_graph.cytoscape_data(partial_G))}))
    
    communities = get_communities(G, 'weight', seed=42)
    cy_data = json.dumps(nx.readwrite.json_graph.cytoscape_data(G))
    response = {
        'cy_data': cy_data,
        'communities': communities
    }
    return json.dumps(response)