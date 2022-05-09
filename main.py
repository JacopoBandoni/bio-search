from pubmad import get_graph, display_graph

G = get_graph('diabetes', max_publications=1, use_biobert=True, clear_cache=True)

#display_graph(G, hide_isolated_nodes=True)