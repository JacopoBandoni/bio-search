import networkx as nx
from pubmad.utils import download_articles, extract_entities, extract_naive_relations, extract_biobert_relations
from pubmad.types import Article, Entity
from typing import List, Tuple
from datetime import datetime
import time

def get_graph(query: str, max_publications: int = 100, start_year: int = 1800, end_year: int = datetime.now().year, use_biobert: bool = True, source: str = 'abstract', save_graph: bool = True) -> nx.Graph:
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
        
        Returns
            nx.Graph (networkx.Graph): A networkx graph containing relationships between genes and diseases.
    '''
    # Download the articles using pymed
    articles: List[Article] = download_articles(query, start_year, end_year, max_publications)

    G = nx.Graph()

    i = 1
    N = len(articles)

    for article in articles:
        if use_biobert == False:
            entities: List[Entity] = extract_entities(article, source)

            relations: List[Tuple[Entity, Entity]] = extract_naive_relations(entities)
        else:
            entities, relations = extract_biobert_relations(article, source)
        
        # Add the entities to the graph as nodes
        for entity in entities:
            G.add_node(entity.mesh_id[0], mention=entity.mention, type=entity.type)

        # Add the relations to the graph as edges
        for src, dst in relations:
            G.add_edge(src.mesh_id[0], dst.mesh_id[0])

        print(f'{i}/{N}')
        if i % 100 == 0:
            print('Sleeping for 100 seconds')
            time.sleep(100)
        i += 1

    # Save the graph in cytoscape format
    if save_graph == True:
        file_name = f"{query}_{start_year}_{end_year}_{max_publications}_{source}_{'BioBert' if use_biobert else 'Naive'}.graphml"
        nx.write_graphml(G, file_name)
        print("Graph saved in:", file_name)
    
    return G