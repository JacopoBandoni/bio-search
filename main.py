from pubmad import get_graph, get_communities, html_graph, display_graph

G = get_graph('diabetes', max_publications=2, use_biobert=True, clear_cache=True)

communities = get_communities(G, "weight")

html_graph(G, "diabetes", communities)