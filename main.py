from dis import dis
from pubmad import *

G = get_graph('diabetes', 10)

insulin = filter_by_name(G, 'insulin')[0]

paths = search_paths_to_category(G, insulin, 'drug')

new_G = get_graph_from_paths(G, paths)

display_graph(G, show=False)

display_graph(new_G)