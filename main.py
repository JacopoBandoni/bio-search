from pubmad import get_graph, display_graph

G = get_graph('diabetes', max_publications=1000, use_biobert=True)

display_graph(G, hide_isolated_nodes=True)