import os
import networkx as nx
import networkx.algorithms.community as nx_comm
from pubmad.utils import download_articles, extract_entities, extract_naive_relations, extract_biobert_relations
from pubmad.types import Article, Entity
from typing import List, Tuple
from datetime import datetime
import time
from tqdm import tqdm

def get_communities(G: nx.Graph, weight_label: str = 'weight', seed: int = 42) -> List[List[str]]:
    comm = nx_comm.louvain_communities(G, weight_label, seed=seed)
    comm = [list(c) for c in comm if len(list(c)) > 1]
    return comm

def get_graph(query: str, max_publications: int = 10, start_year: int = 1800, end_year: int = datetime.now().year, use_biobert: bool = True, source: str = 'abstract', save_graph: bool = True, G: nx.Graph = None, clear_cache: bool = False) -> nx.Graph:
    '''
    Returns a networkx graph containing relationships between genes and diseases.
    
        Parameters
            query (str): The query to be used to search PubMed.
            max_publications (int): The maximum number of publications to be used.
            start_year (int): The start year to be used.
            end_year (int): The end year to be used.
            use_biobert (bool): Whether to use BioBERT or not.
            source (str): The source to be used. Can be 'abstract' or 'full_text'. Defaults to 'abstract'.
            save_graph (bool): Whether to save the graph or not. Defaults to True.
            G (nx.Graph): The graph to be used. Defaults to an empty graph.
            clear_cache (bool): Whether to clear the cache or not. Defaults to False.
            
        Returns
            nx.Graph (networkx.Graph): A networkx graph containing relationships between genes and diseases.
    '''
    if G is None:
        G = nx.Graph()
        G.clear()

    # Download the articles using pymed
    articles: List[Article] = download_articles(title=query, start_year=start_year, end_year=end_year, max_results = max_publications)

    bern_calls_counter = 1
    i = 0 # current article
    N = len(articles)
    start = time.time()

    for i in tqdm(range(0,len(articles))):
        article = articles[i]
        if use_biobert == False:
            entities: List[Entity] = extract_entities(article, source)

            relations: List[Tuple[Entity, Entity, float]] = extract_naive_relations(entities)
        else:
            # if bern_calls_counter % 100 == 0:
            #     # We made 100 calls, we must wait for 100s - start to not get banned
            #     print("Sleeping for {}s".format(110 - (time.time() - start)))
            #     time.sleep(max(110 - (time.time() - start), 20))
            #     start = time.time()
            entities, relations, used_cache = extract_biobert_relations(article, source, clear_cache)
            if not used_cache:
                bern_calls_counter += 1
                elapsed = time.time() - start
                if elapsed < 1:
                    time.sleep(1.1 - elapsed)
                start = time.time()
        
        # Add the entities to the graph as nodes
        for entity in entities:
            # Search if there is already a node with the same mesh_id
            if entity.mesh_id[0] not in G.nodes:
                G.add_node(entity.mesh_id[0], mention=entity.mention, type=entity.type, pmid=entity.pmid)
            else:
                # If there is already a node with the same mesh_id, add the mention to the node
                #G.nodes[entity.mesh_id[0]]['mention'] += ' ' + entity.mention
                G.nodes[entity.mesh_id[0]]['pmid'] += ',' + entity.pmid

        # Add the relations to the graph as edges
        for src, dst, weight in relations:
            G.add_edge(src.mesh_id[0], dst.mesh_id[0], weight=weight)
        i += 1
        #print(f'Processing article {i}/{N}')

    # Save the graph in cytoscape format
    if save_graph == True:
        #create output directory if not exist
        if not os.path.exists("output"):
            # Create a new directory because it does not exist 
            os.makedirs("output")
            print("The new directory is created!")
 
        file_name = f"output/{query}_{start_year}_{end_year}_{max_publications}_{source}_{'BioBert' if use_biobert else 'Naive'}.graphml"
        
        
        nx.write_graphml(G, file_name)
        print("Graph saved in:", file_name)
    
    return G

def add_nodes(graph: nx.Graph, query: str, max_publications: int = 10, start_year: int = 1800, end_year: int = datetime.now().year, use_biobert: bool = True, source: str = 'abstract', save_graph: bool = True) -> nx.Graph:
    return get_graph(query, max_publications, start_year, end_year, use_biobert, source, save_graph, graph)