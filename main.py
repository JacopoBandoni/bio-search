from pubmad import get_graph, get_communities, html_graph, display_graph
import json
import networkx as nx

G = get_graph('diabetes', max_publications=10, use_biobert=True, clear_cache=False)

display_graph(G)