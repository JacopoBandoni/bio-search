from pubmad import get_graph, display_graph

G = get_graph('diabetes', max_publications=300, use_biobert=True, clear_cache=False)

display_graph(G, hide_isolated_nodes=True)