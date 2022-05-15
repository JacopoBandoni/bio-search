import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple
import requests
import networkx as nx
from itertools import product
from pubmad.types import Article, Entity
import matplotlib.pyplot as plt
import os
from nltk.tokenize import sent_tokenize
import pickle
from pathlib import Path
import time
from Bio import Entrez, Medline
from pyvis.network import Network
import re
import random
import numpy as np
import IPython
from math import ceil

import nltk
nltk.download('punkt', quiet=True)

# Number of sequences per batch
MAX_SEQ_PER_BATCH = 2

print("Loading models...")
chemprot_model = AutoModelForSequenceClassification.from_pretrained(
    "pier297/autotrain-chemprot-re-838426740")
chemprot_tokenizer = AutoTokenizer.from_pretrained(
    "pier297/autotrain-chemprot-re-838426740")

rel_tokenizer = AutoTokenizer.from_pretrained(
    "JacopoBandoni/BioBertRelationGenesDiseases")

rel_model = AutoModelForSequenceClassification.from_pretrained(
    "JacopoBandoni/BioBertRelationGenesDiseases")
print('Finished loading models.')

chemprot_model.eval()
rel_model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rel_model = rel_model.to(device)
chemprot_model = chemprot_model.to(device)

# logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
# or
# disable WARNING, INFO and DEBUG logging everywhere
logging.disable(logging.WARNING)


def download_articles(title: str, start_year: int, end_year: int, max_results: int = 100, author: str = '', sort_by: str = 'relevance') -> List[Article]:
    """
    Download articles from PubMed using BioPython library.
    Args:
        query (str): The query to search for.
        start_year (int): The start year to search for.
        end_year (int): The end year to search for.
        max_results (int): The maximum number of results to return.
        author (str): The author to search for, leave empty to search for all authors.
        sort_by (str): The sort order to use. Can be 'relevance' to retrieve the most relevant results, or 'date' to sort by publication date. Defaults to 'relevance'.
    Returns:
        List[Article] A list of articles.
    """
    current_path = Path(os.getcwd()) / 'cache'

    if not os.path.exists(current_path):
        os.mkdir(current_path)

    file_name = '{}_{}_{}_{}.txt'.format(
        title, start_year, end_year, max_results)
    if os.path.exists(current_path / file_name):
        print("Loading from cache...")
        # TODO: clear_cache is not implemented yet
        with open(current_path / file_name, 'rb') as f:
            return pickle.load(f)

    Entrez.email = 'pubmadbiosearch@gmail.com'

    if author != "":
        query = '(' + title + ') AND (' + author + '[Author])'
    else:
        query = title

    handle = Entrez.esearch(db='pubmed',
                            sort=sort_by,
                            retmax=max_results,
                            retmode='xml',
                            datetype='pdat',
                            mindate=str(start_year),
                            maxdate=str(end_year),
                            term=query)
    results = Entrez.read(handle)
    id_list = results['IdList']
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', rettype="medline",
                           id=ids, retmode='text')
    results = Medline.parse(handle)
    results = list(results)
    articles = []
    for article in results:
        abstract = article.get("AB", "?")
        title = article.get("TI", "?")
        if abstract != "?" and title != "?":
            if abstract.find("This article has been withdrawn at the request of the author(s) and/or editor") == -1 and \
               abstract.find("An amendment to this paper has been published and can be accessed via a link at the top of the paper") == -1 and \
               abstract.find("This corrects the article") == -1 and abstract.find("A Correction to this paaper has been published") == -1:
                articles.append(Article(title=title, abstract=abstract,
                                        pmid=article['PMID'], full_text='', publication_data=None))

    print("Found {} articles".format(len(articles)))

    # save articles to pickle file
    with open(current_path / file_name, 'wb') as f:
        pickle.dump(articles, f)
    return articles


def extract_entities_pmids(pmids: List[str]) -> List[List[Entity]]:
    """Extract entities from one or more articles identified by the pmids provided

    Args:
        pmids (List[str]): list of pmids to extract entities from

    Returns:
        List[List[Entity]]: list of entities for each article
    """

    if len(pmids) == 0:
        return []

    # Query BERN2 server for entities.
    extracted = False
    bern_result = None

    i = 0

    while not extracted and i < 3:
        try:
            bern_result = query_pmid(pmids)
            extracted = True
        except Exception as e:
            print("Error querying BERN2 server")
            print("Sleep for 3 seconds and try again to not get banned by BERN2")
            time.sleep(3)
            i += 1
            continue

    entities = []

    # Parse the result
    for i, article in enumerate(bern_result):
        entities_per_article = []
        for entity in article["annotations"]:
            if (entity['obj'] == 'disease' or entity['obj'] == 'gene' or entity['obj'] == 'drug'):
                new_entity = Entity(mesh_id=entity['id'], mention=entity['mention'], type=entity['obj'],
                                    prob=entity['prob'], span_begin=entity['span']['begin'], span_end=entity['span']['end'], pmid=pmids[i])
                entities_per_article.append(new_entity)
        entities.append(entities_per_article)

    return entities


def extract_entities(article: Article, source: str = 'abstract') -> List[Entity]:
    """
    Extract entities from an article using BERN2 (online).

    Args:
        article (Article): The article to extract entities from. 
        source (str): The source to extract entities from. Can be 'abstract' or 'full_text'. Defaults to 'abstract'.

    Returns:
        List[Entity]: A list of entities.
    """
    # We call bern2 server for ner
    text = ''
    if source == 'abstract':
        text = article.abstract
    elif source == 'full_text':
        text = article.full_text
    else:
        raise ValueError("Invalid source: {}".format(source))

    # Query BERN2 server for entities.
    extracted = False
    bern_result = None

    i = 0

    while not extracted and i < 3:
        try:
            bern_result = query_plain(text)
            extracted = True
        except Exception as e:
            print("Error querying BERN2 server")
            print("Sleep for 3 seconds and try again to not get banned by BERN2")
            time.sleep(3)
            i += 1
            continue

    if i == 3:
        return []

    # Parse the result.
    annotations = bern_result['annotations']
    entities = []
    for entity in annotations:
        if (entity['obj'] == 'disease' or entity['obj'] == 'gene' or entity['obj'] == 'drug'):
            pmid = article.pmid
            if type(article.pmid) == list:
                pmid = article.pmid[0]
            new_entity = Entity(mesh_id=entity['id'], mention=entity['mention'], type=entity['obj'],
                                prob=entity['prob'], span_begin=entity['span']['begin'], span_end=entity['span']['end'], pmid=pmid)
            entities.append(new_entity)

    return entities


def display_graph(graph: nx.Graph, hide_isolated_nodes: bool = True, show: bool = True):
    """
    Display a graph.

    Args:
        graph (nx.Graph): The graph to display.
        hide_isolated_nodes (bool): Whether to hide isolated nodes. Defaults to True.
        show (bool): Whether to show the graph. Defaults to True. Otherwise it shows the window without stopping the program.
    """
    plt.figure()
    if hide_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))

    gene_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'gene']
    disease_nodes = [n for n, d in graph.nodes(
        data=True) if d['type'] == 'disease']
    other_nodes = [n for n, d in graph.nodes(
        data=True) if d['type'] != 'gene' and d['type'] != 'disease']

    pos = nx.spring_layout(graph, 100)

    weights = nx.get_edge_attributes(graph, 'weight')

    nx.draw_networkx_nodes(graph, pos, nodelist=gene_nodes,
                           node_color='r', node_size=100, alpha=0.8, label='gene')
    nx.draw_networkx_nodes(graph, pos, nodelist=disease_nodes,
                           node_color='b', node_size=100, alpha=0.8, label='disease')
    nx.draw_networkx_nodes(graph, pos, nodelist=other_nodes,
                           node_color='g', node_size=100, alpha=0.8, label='drug')

    nx.draw_networkx_labels(graph, pos, labels={
                            n: d['mention'] for n, d in graph.nodes(data=True)}, font_size=10)

    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    plt.axis('off')
    plt.legend()
    if show:
        plt.show()
    else:
        plt.draw()


def extract_naive_relations(entities: List[Entity]) -> List[Tuple[Entity, Entity, float]]:
    '''
    Extract relations from a list of entities.

    Args:
        entities (List[Entity]): A list of entities.

    Returns:
        List[Tuple[Entity, Entity, float]]: A list of relations.
    '''
    # Connect all the entities to each other.
    relations = []
    for entity1, entity2 in product(entities, entities):
        if entity1.mesh_id != entity2.mesh_id:
            # 1 meaning that the relation is present.
            relations.append((entity1, entity2, 1))
    return relations


def extract_biobert_relations(article: Article, source: str = 'abstract', clear_cache: bool = False) -> Tuple[List[Entity], List[Tuple[Entity, Entity]], bool]:
    """
    Extract the entities from the article using BERN2 (online).
    Then it calls BioBERT to extract relations between genes and diseases.

    Args:
        article (Article): The article to extract entities from.
        source (str): The source to extract entities from. Can be 'abstract' or 'full_text'. Defaults to 'abstract'.
        clear_cache (bool): Whether to clear the BioBERT cache. Defaults to False.

    Returns:
        Tuple[List[Entity], List[Tuple[Entity, Entity]], bool]: A tuple of entities, relations and whether the cache was used.

    """
    file_name = str(article.pmid)[:128] + '.txt'
    file_name = file_name.replace('\n', '_')
    path = Path(os.getcwd()) / 'cache'

    if not os.path.exists(path / 'entities'):
        # create directory
        os.mkdir(path / 'entities')

    if not os.path.exists(path / 'relations'):
        # create directory
        os.mkdir(path / 'relations')

    if os.path.exists(path / 'entities' / file_name) and os.path.exists(path / 'relations' / file_name):
        if clear_cache:
            print("Clearing entities and relations cache for PMID {}".format(
                article.pmid))
            os.remove(path / 'entities' / file_name)
            os.remove(path / 'relations' / file_name)
        else:
            with open(path / 'entities' / file_name, 'rb') as f:
                entities = pickle.load(f)
            with open(path / 'relations' / file_name, 'rb') as f:
                relations = pickle.load(f)
            return entities, relations, True

    text = ''
    if source == 'abstract':
        text = article.abstract
    elif source == 'full_text':
        text = article.full_text
    else:
        raise ValueError("Invalid source: {}".format(source))

    entities = extract_entities(article)

    span_sentences = tokenize_into_sentences(article, source)

    # divide entities in gene and disease entities
    gene_entities = []
    disease_entities = []
    drug_entities = []
    for entity in entities:
        if entity.type == "disease":
            disease_entities.append(entity)
        elif entity.type == "gene":
            gene_entities.append(entity)
        elif entity.type == "drug":
            drug_entities.append(entity)

    relations = []
    chemprot_batch = []
    biobert_batch = []
    for gene_idx, gene_entity in enumerate(gene_entities):
        # extract gene-drug relations using fine-tuned bert on chemprot
        for drug_idx, drug_entity in enumerate(drug_entities):
            # find the sentence that contains the gene and disease
            sentence_index_gene = find_entity(gene_entity, span_sentences)
            sentence_index_drug = find_entity(drug_entity, span_sentences)

            masked_text = ''
            if gene_entity.span_begin < drug_entity.span_begin:
                # wrap the gene_entity in << >>
                # and the drug with [[ ]]
                masked_text = text[span_sentences[sentence_index_gene][0]:gene_entity.span_begin] + "<< " + text[gene_entity.span_begin:gene_entity.span_end] + " >>" + \
                    text[gene_entity.span_end:drug_entity.span_begin] + "[[ " + text[drug_entity.span_begin:drug_entity.span_end] + \
                    " ]]" + \
                    text[drug_entity.span_end:span_sentences[sentence_index_drug][1]]
            else:
                # wrap the drug_entity in << >>
                # and the gene with [[ ]]
                masked_text = text[span_sentences[sentence_index_drug][0]:drug_entity.span_begin] + "<< " + text[drug_entity.span_begin:drug_entity.span_end] + " >>" + \
                    text[drug_entity.span_end:gene_entity.span_begin] + "[[ " + text[gene_entity.span_begin:gene_entity.span_end] + \
                    " ]]" + \
                    text[gene_entity.span_end:span_sentences[sentence_index_gene][1]]

            chemprot_batch.append(
                {'gene_idx': gene_idx, 'drug_idx': drug_idx, 'masked_text': masked_text})

        # extract gene-disease relations using biobert
        for disease_idx, disease_entity in enumerate(disease_entities):
            # find the sentence that contains the gene and disease
            sentence_index_gene = find_entity(gene_entity, span_sentences)
            sentence_index_disease = find_entity(
                disease_entity, span_sentences)

            masked_text = ''
            if gene_entity.span_begin < disease_entity.span_begin:
                masked_text = text[span_sentences[sentence_index_gene][0]:gene_entity.span_begin] + "@GENE$" + text[gene_entity.span_end:
                                                                                                                    disease_entity.span_begin] + "@DISEASE$" + text[disease_entity.span_end:span_sentences[sentence_index_disease][1]]
            else:
                masked_text = text[span_sentences[sentence_index_disease][0]:disease_entity.span_begin] + "@DISEASE$" + \
                    text[disease_entity.span_end:gene_entity.span_begin] + "@GENE$" + \
                    text[gene_entity.span_end:span_sentences[sentence_index_gene][1]]

            biobert_batch.append(
                {'gene_idx': gene_idx, 'disease_idx': disease_idx, 'masked_text': masked_text})

    # Predict the relations using biobert
    if len(biobert_batch) > 0:
        for batch_idx in range(0, len(biobert_batch), MAX_SEQ_PER_BATCH):
            batch = biobert_batch[batch_idx:batch_idx + MAX_SEQ_PER_BATCH]
            masked_texts = [x['masked_text'] for x in batch]
            tok_texts = rel_tokenizer(
                masked_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = rel_model(**tok_texts)
            class_logits = outputs["logits"].detach().cpu().numpy()
            # len x 2
            # apply softmax to get the probabilities
            class_probs = np.exp(
                class_logits) / np.sum(np.exp(class_logits), axis=1, keepdims=True)
            for i in range(len(class_logits)):
                if class_probs[i][0] > 0.5:
                    j = i + batch_idx
                    gene = gene_entities[biobert_batch[j]['gene_idx']]
                    disease = disease_entities[biobert_batch[j]['disease_idx']]
                    relations.append((gene, disease, float(class_probs[i][0])))

            # free gpu memory
            del tok_texts

    # # Predict the relations using chemprot
    if len(chemprot_batch) > 0:
        for batch_idx in range(0, len(chemprot_batch), MAX_SEQ_PER_BATCH):
            batch = chemprot_batch[batch_idx:batch_idx + MAX_SEQ_PER_BATCH]
            masked_texts = [x['masked_text'] for x in batch]
            tok_texts = chemprot_tokenizer(
                masked_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = chemprot_model(**tok_texts)
            class_logits = outputs["logits"].detach().cpu().numpy()
            # len x 13
            # apply softmax to get the probabilities
            class_probs = np.exp(
                class_logits) / np.sum(np.exp(class_logits), axis=1, keepdims=True)
            for i in range(len(class_logits)):
                # if 1 class is above 0.5, then we consider it as a relation
                max_p = np.max(class_probs[i])
                if max_p > 0.5:
                    j = i + batch_idx
                    gene = gene_entities[chemprot_batch[j]['gene_idx']]
                    drug = drug_entities[chemprot_batch[j]['drug_idx']]
                    relations.append((gene, drug, float(max_p)))

            # free gpu memory
            del tok_texts

    # save entities and relations to file with pickle
    with open(path / 'entities' / file_name, 'wb') as f:
        pickle.dump(entities, f)
    with open(path / 'relations' / file_name, 'wb') as f:
        pickle.dump(relations, f)

    return entities, relations, False


def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    """
    Query the BERN2 server for plain text.
    Args:
        text (str): the text to be annotated
        url (str): the url of the BERN2 server.
    """
    # Limit the text at 5000 characters
    if len(text) > 5000:
        text = text[:5000]

    result = requests.post(url, json={'text': text})

    if result.status_code != 200:
        print("Error: {}".format(result.status_code))
        raise Exception("Error: {}".format(result.status_code))

    return result.json()


def query_pmid(pmids, url="http://bern2.korea.ac.kr/pubmed"):
    """query the BERN2 server for pmids"""
    return requests.get(url + "/" + ",".join(pmids)).json()


def tokenize_into_sentences(article: Article, source: str = 'abstract') -> List[List[int]]:
    """tokenize the text into sentences returning the index of begin and end of each sentence

    Args:
        article (Article): article to tokenize.
        source (str): The source of the text.

    Returns:
        List: A list of couples containing the begin and end of each sentence.
    """
    text = ''
    if source == 'abstract':
        text = article.abstract
    elif source == 'full_text':
        text = article.full_text
    else:
        raise ValueError("Invalid source: {}".format(source))

    sentences = sent_tokenize(text)
    span_sentences = []
    # for each sentence, put the start and end index of each sentence
    start = 0
    for sentence in sentences:
        span_sentences.append([start, start + len(sentence)])
        start += len(sentence) + 1
    return span_sentences


def find_entity(entity: Entity, span_sentences: List[List[int]]) -> int:
    """find in which sentence the entity is

    Args:
        entity (Entity): entity to find
        span_sentences (List): A list of couples containing the index of the begin and end of each sentence.

    Returns:
        int: index of the sentence containing the entity
    """
    for i in range(len(span_sentences)):
        if span_sentences[i][0] <= entity.span_begin and entity.span_end <= span_sentences[i][1]:
            return i
    return -1


def add_title_node(G, n, d):
    result_html = f"""
  <h3>{d['mention']}</h3>
  """
    # articles:
    #  <ul>
    # """
    #pmids = re.split("[,\\n]", d['pmid'])
    # for pmid in pmids:
    #  result_html += f"<li>{pmid}</li>"
    #result_html += "</ul>"
    return result_html


def add_title_edge(G, edge):
    # find intersection between pmids of the two nodes
    node1 = dict(G.nodes(data=True))[edge[0]]
    node2 = dict(G.nodes(data=True))[edge[1]]
    n1_pmids = re.split(",", node1['pmid'])
    n2_pmids = re.split(",", node2['pmid'])
    intersection = list(set(n1_pmids) & set(n2_pmids))

    # create html
    result_html = f"""<p>common articles between <strong>{node1['mention']}</strong> and <strong>{node2['mention']}</strong></p>
  
  <ul>"""
    for pmid in intersection:
        result_html += f"<li>{pmid}</li>"
    result_html += "</ul>"
    return result_html


def _html_graph_communities(G, communities, net, colors):
    for i, community in enumerate(communities):
        for node_id in community:
            node = net.get_node(node_id)
            node_type = dict(G.nodes(data=True))[node_id]['type']
            node['color'] = colors[i]

            if (node_type == 'gene'):
                node['shape'] = 'dot'
            elif (node_type == 'disease'):
                node['shape'] = 'diamond'
            elif (node_type == 'drug'):
                node['shape'] = 'star'


def _find_community(node_id, communities):
    """
        Find the index of the community containing the node.
    """
    for i, community in enumerate(communities):
        if node_id in community:
            return i


def html_mark_subgraph(G, subG, name="nodes", hide_isolated_nodes=True):
    """
        Create a html file, named <name>, containing the graph G with the subgraph subG marked.

        Args:
            G (networkx.Graph): The graph to be displayed.
            subG (networkx.Graph): The subgraph to be marked.
            name (str): The name of the html file. Default is "nodes".
            hide_isolated_nodes (bool): If True, the isolated nodes are not displayed. Default is True.
    """

    if hide_isolated_nodes:
        G.remove_nodes_from(list(nx.isolates(G)))
        subG.remove_nodes_from(list(nx.isolates(subG)))

    net = Network('500px', '600px', notebook=True)

    for n, d in G.nodes(data=True):
        color = '#cccccc80'
        if n not in subG.nodes():
            net.add_node(n, d['mention'], color='#cccccc80',
                title=add_title_node(G, n, d))

    for n, d in subG.nodes(data=True):
        color = '#cccccc80'
        if d['type'] == 'gene':
            color = '#0DA3E4'
        elif d['type'] == 'disease':
            color = '#bf4d2d'
        elif d['type'] == 'drug':
            color = '#5ef951'
        net.add_node(n, d['mention'], color=color,
            title=add_title_node(G, n, d))

    # add edges not in subG
    for edge in G.edges(data='True'):
        weight = G.get_edge_data(edge[0], edge[1])['weight']
        net.add_edge(edge[0], edge[1], value=weight,
            color='#cccccc70', title=add_title_edge(G, edge))

    # add edges in subG
    for edge in subG.edges(data='True'):
        weight = G.get_edge_data(edge[0], edge[1])['weight']
        net.add_edge(edge[0], edge[1], value=weight,
            color='#F0EB5A', title=add_title_edge(subG, edge))

    net.show_buttons(filter_=['physics'])
    net.show(name + ".html")
    IPython.display.HTML(filename=Path(os.getcwd())/(name + ".html"))


def html_graph(G, name="nodes", communities=None, hide_isolated_nodes=True):
    """
        Create an html file, named <name>, representing the graph G using vis.js.

        Args:
            G (NetworkX graph): The graph to be displayed.
            name (str): The name of the file to be created. Default is "nodes".
            communities (List): A list of communities to be displayed. If None, no communities are displayed. Default is None.
            hide_isolated_nodes (bool): If True, hide isolated nodes. Default is True.

        Returns:
            A pyvis network object and writes the file to the current directory.
    """

    if hide_isolated_nodes:
        G.remove_nodes_from(list(nx.isolates(G)))

    net = Network('500px', '600px', notebook=True)

    for n, d in G.nodes(data=True):
        color = '#ccc'
        if d['type'] == 'gene':
            color = '#0DA3E4'
        elif d['type'] == 'disease':
            color = '#bf4d2d'
        elif d['type'] == 'drug':
            color = '#5ef951'
        net.add_node(n, d['mention'], color=color,
                     title=add_title_node(G, n, d))

    # if communities is not None and len(communities) > 0
    colors = []
    if communities is not None and len(communities) > 0:
        # generate random color for each community
        colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) + "90"
                  for i in range(len(communities))]
        _html_graph_communities(G, communities, net, colors)

    # add edges
    for edge in G.edges(data='True'):
        weight = G.get_edge_data(edge[0], edge[1])['weight']
        if communities is not None and len(communities) > 0:
            # find the community of the two nodes
            community_node1_index = _find_community(edge[0], communities)
            if edge[1] in communities[community_node1_index]:
                net.add_edge(edge[0], edge[1], value=weight,
                             color=colors[community_node1_index], title=add_title_edge(G, edge))
            else:
                net.add_edge(edge[0], edge[1], value=weight,
                             color='#F0EB5A', title=add_title_edge(G, edge))
        else:
            net.add_edge(edge[0], edge[1], value=weight,
                         color='#F0EB5A', title=add_title_edge(G, edge))

    net.show_buttons(filter_=['physics'])
    net.show(name + ".html")
    IPython.display.HTML(filename=Path(os.getcwd())/(name + ".html"))

    return net


def filter_by_centrality(G, percentage_threshold: float = 0.1):
    '''
    Filter the graph by mantaining the 'percentage_threshold' of the nodes with the highest centrality.
    Args:
        G: networkx graph
        percentage_threshold: float between 0 and 1. The percentage of most central nodes to keep.
    '''
    mesh_id_to_degree = dict()
    for node_id in G.nodes():
        mesh_id_to_degree[node_id] = G.degree(node_id)

    sorted_mesh_ids = sorted(
        mesh_id_to_degree, key=mesh_id_to_degree.get, reverse=True)

    # take only the nodes with the highest centrality
    nodes_to_keep = sorted_mesh_ids[:ceil(
        len(sorted_mesh_ids)*percentage_threshold)]

    return G.subgraph(nodes_to_keep).copy()


def filter_by_name(G, name: str):
    """
    Filter the graph by mantaining the nodes with the given name.
    Args:
        G: networkx graph
        name: str. The name of the nodes to keep.
    Returns:
        A list of nodes that include the given name.
    """
    nodes = []
    for n, d in G.nodes(data=True):
        if name in d['mention']:
            d['mesh_id'] = n
            nodes.append(d)

    return nodes


def filter_by_category(G, category: str, max_number_of_nodes: int = None, sort_by: str = 'degree'):
    '''
    Returns the nodes with the type category
    Args:
        G (Networkx Graph): graph to filter
        category (str): category to filter by: 'gene', 'disease', 'drug'
        max_number_of_nodes (int): maximum number of nodes to return. If None, all nodes are returned
        sort_by (str): sort by 'degree' or 'name'
    Returns:
        A list of nodes with the given category
    '''
    if category not in ['gene', 'disease', 'drug']:
        raise ValueError("category must be one of 'gene', 'disease', 'drug'")

    nodes = []
    for n, d in G.nodes(data=True):
        if d['type'] == category:
            d['mesh_id'] = n
            nodes.append(d)

    if sort_by == 'degree':
        nodes = sorted(nodes, key=lambda node: G.degree(
            node['mesh_id']), reverse=True)
    elif sort_by == 'name':
        nodes = sorted(nodes, key=lambda node: node['mention'])

    if max_number_of_nodes is not None:
        nodes = nodes[:max_number_of_nodes]

    return nodes


def expand_from_node(G, node, max_distance):
    ''' 
    Returns the subgraph that starting from the node id link all the node with distance k from id
    Args:
        G (Networkx Graph): graph to filter
        node (str): node id
        max_distance (int): maximum distance to expand
    Returns:
        A new graph containing the nodes with distance k from node
    '''
    if type(node) == List:
        node = node[0]
        print("node is a list, taking the first element")
        print("Otherwise use 'expand_from_nodes'")
    node_list = [node['mesh_id']]

    for node, succ in nx.bfs_successors(G, node['mesh_id'], depth_limit=max_distance):
        node_list += list(succ)

    return G.subgraph(node_list).copy()


def expand_from_nodes(G, nodes, max_distance: int):
    '''
    Returns the subgraph that starting from the node id link all the node with distance k from id 
    Args:
        G (Networkx Graph): graph to filter
        nodes (list): list of nodes mesh_id
        max_distance (int): maximum distance to expand
    Returns:
        A new graph containing the nodes with distance k from node
    '''
    if type(nodes[0]) == dict:
        node_id_list = [node['mesh_id'] for node in nodes]

        node_list = [node['mesh_id'] for node in nodes]
    elif type(nodes[0]) == str:
        node_id_list = nodes
        node_list = nodes
    else:
        raise ValueError("nodes must be a list of mesh_id or a list of node objects")

    for node_id in node_id_list:
        for _, succ in nx.bfs_successors(G, node_id, depth_limit=max_distance):
            node_list += list(succ)

    return G.subgraph(node_list).copy()


def get_common_articles(node1, node2):
    '''
    Returns the articles that both nodes mention
    Args:
        node1 (dict): node object
        node2 (dict): node object
    Returns:
        A list of articles
    '''
    articles_node1 = node1['pmid']
    articles_node2 = node2['pmid']
    return list(set(articles_node1.split(',')).intersection(articles_node2.split(',')))


def search_path(G, from_node, to_node):
    '''
    Returns the path between two nodes in the graph
    Args:
        G (Networkx Graph): graph to filter
        from_node (str): node id
        to_node (str): node id
    Returns:
        A list of nodes that form the path
    '''
    try:
        return nx.shortest_path(G, source=from_node['mesh_id'], target=to_node['mesh_id'])
    except nx.NetworkXNoPath:
        return []


def search_paths_to_category(G, from_node, to_category: str):
    '''
    Returns the paths from 'from_node' to nodes with the type 'to_category'
    Args:
        G (Networkx Graph): graph to filter
        from_node (str): node id
        to_category (str): to_category to filter by: 'gene', 'disease', 'drug'
    Returns:
        A list of paths
    '''
    if to_category not in ['gene', 'disease', 'drug']:
        raise ValueError("to_category must be one of 'gene', 'disease', 'drug'")

    targets_nodes = filter_by_category(G, to_category)
    paths = []
    for target_node in targets_nodes:
        paths.append(search_path(G, from_node, target_node))

    # remove empty paths
    paths = [path for path in paths if len(path) > 0]

    return paths


def get_graph_from_path(G, path):
    '''
    Returns the subgraph that contains the nodes in the path
    Args:
        G (Networkx Graph): graph to filter
        path (list): list of nodes mesh_id
    Returns:
        A new graph containing the nodes in the path
    '''
    return G.subgraph(path).copy()


def get_graph_from_paths(G, paths):
    '''
    Returns the subgraph that contains the nodes in the paths
    Args:
        G (Networkx Graph): graph to filter
        paths (list): list of paths
    Returns:
        A new graph containing the nodes in the paths
    '''
    nodes = [node for path in paths for node in path]
    return G.subgraph(nodes).copy()