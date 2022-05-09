from pubmad import get_graph, get_communities, html_graph, display_graph

G = get_graph('diabetes', max_publications=18, use_biobert=True, clear_cache=False)

communities = get_communities(G, "weight")

html_graph(G, "diabetes", communities)
#display_graph(G, hide_isolated_nodes=True)