import enum
import torch
from typing import List, Optional, Tuple
from pymed import PubMed 
import json 
import requests
import networkx as nx
from datetime import datetime
from itertools import product
from pubmad.types import Article, Entity
import matplotlib.pyplot as plt
import os
import random
import string
from multiprocessing import Pool, cpu_count
import tqdm
from nltk.tokenize import sent_tokenize
from Bio import Entrez, Medline
import pickle
from pathlib import Path
import time
from joblib import Parallel, delayed
import copy

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

rel_tokenizer = AutoTokenizer.from_pretrained("./pubmad/biobert/relation-extraction")
rel_model = AutoModelForSequenceClassification.from_pretrained("./pubmad/biobert/relation-extraction")

rel_model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rel_model = rel_model.to(device)

rel_pipe = pipeline(task='text-classification', model=rel_model, tokenizer=rel_tokenizer, device=0)


def download_articles(query: str, start_year: int, end_year: int, max_results: int = 100, author: str = '') -> List[Article]:
    """
    Download articles from PubMed.

    Args:
        query (str): The query to search for.
        start_year (int): The start year to search for.
        end_year (int): The end year to search for.
        max_results (int): The maximum number of results to return.
        author (str): The author to search for, leave empty to search for all authors.

    Returns:
        List[Article] A list of articles.
    """
    current_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'cache'
    file_name = '{}_{}_{}_{}.txt'.format(query, start_year, end_year, max_results)
    if os.path.exists(current_path / file_name):
        print("Loading from cache...")
        with open(current_path / file_name, 'rb') as f:
            return pickle.load(f)
    
    pubmed = PubMed(tool="PubMad", email="pubmadbiosearch@gmail.com")
    results = pubmed.query('(' + query + '[Title]) AND ' + '(' + author + '[Author])' + ' AND ' + '(("' + str(start_year) + '"[Date - Create] : "' +
                           str(end_year) + '"[Date - Create]))', max_results=max_results)
                           
    articles = []
    for article in results:
        article = article.toJSON()
        article = json.loads(article)
        if article['abstract'] == None:
            # TODO: Handle this case. (either by filtering within the query or by skipping the article)
            continue
        articles.append(Article(title=article['title'], abstract=article['abstract'], 
                                pmid=article['pubmed_id'], full_text='', publication_data=datetime.strptime(article['publication_date'][:4], '%Y')))

    # save articles to pickle file
    with open(current_path / file_name, 'wb') as f:
        pickle.dump(articles, f)
    return articles


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
    while not extracted:
        try:
            bern_result = query_plain(text)
            extracted = True
        except Exception as e:
            print("Error querying BERN2 server")
            print("Sleep for 10 seconds and try again to not get banned by BERN2")
            time.sleep(10)
            continue

    # Parse the result.
    annotations = bern_result['annotations']
    entities = []
    for entity in annotations:
        if (entity['obj'] == 'disease' or entity['obj'] == 'gene'):
            new_entity = Entity(mesh_id=entity['id'], mention=entity['mention'], type=entity['obj'], prob=entity['prob'], span_begin=entity['span']['begin'], span_end=entity['span']['end'])
            entities.append(new_entity)

    return entities


def display_graph(graph: nx.Graph, hide_isolated_nodes: bool = True):
    """
    Display a graph.

    Args:
        graph (nx.Graph): The graph to display.
        hide_isolated_nodes (bool): Whether to hide isolated nodes. Defaults to True.
    """
    if hide_isolated_nodes:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    
    gene_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'gene']
    disease_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'disease']
    other_nodes = [n for n, d in graph.nodes(data=True) if d['type'] != 'gene' and d['type'] != 'disease']

    pos = nx.spring_layout(graph)

    nx.draw_networkx_nodes(graph, pos, nodelist=gene_nodes, node_color='r', node_size=100, alpha=0.8, label='gene')
    nx.draw_networkx_nodes(graph, pos, nodelist=disease_nodes, node_color='b', node_size=100, alpha=0.8, label='disease')
    nx.draw_networkx_nodes(graph, pos, nodelist=other_nodes, node_color='g', node_size=100, alpha=0.8, label='other')

    nx.draw_networkx_labels(graph, pos, labels={n: d['mention'] for n, d in graph.nodes(data=True)}, font_size=10)

    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    plt.axis('off')
    plt.legend()
    plt.show()


def extract_naive_relations(entities: List[Entity]) -> List[Tuple[Entity, Entity]]:
    '''
    Extract relations from a list of entities.

    Args:
        entities (List[Entity]): A list of entities.

    Returns:
        List[Tuple[Entity, Entity]]: A list of relations.
    '''
    # Connect all the entities to each other.
    relations = []
    for entity1, entity2 in product(entities, entities):
        if entity1.mesh_id != entity2.mesh_id:
            relations.append((entity1, entity2))
    return relations


# def call_biober_rel(gene_disease_text: Tuple[Entity, Entity, str]):
#     # TODO: Delete this if we don't use cpu cores
#     gene_entity, disease_entity, text = gene_disease_text

#     masked_text = ''
#     if gene_entity.span_begin < disease_entity.span_begin:
#         masked_text = text[:gene_entity.span_begin] + "@GENE$" + text[gene_entity.span_end:disease_entity.span_begin] + "@DISEASE$" + text[disease_entity.span_end:]
#     else:
#         masked_text = text[:disease_entity.span_begin] + "@DISEASE$" + text[disease_entity.span_end:gene_entity.span_begin] + "@GENE$" + text[gene_entity.span_end:]
    
#     try:
#         out = rel_pipe(masked_text)[0]
#     except:
#         return (None, None)
#     if out['label'] == 'LABEL_1':
#         # TODO: Add the score as the probability
#         return (gene_entity, disease_entity)
#     return (None, None)

def extract_biobert_relations(article : Article, source: str = 'abstract') -> Tuple[List[Entity], List[Tuple[Entity, Entity]]]:
    """
    Extract the entities from the article using BERN2 (online).
    Then it calls BioBERT to extract relations between genes and diseases.

    Args:
        article (Article): The article to extract entities from.
        source (str): The source to extract entities from. Can be 'abstract' or 'full_text'. Defaults to 'abstract'.

    Returns:
        List[Tuple[Entity, Entity]]: A list of relations.

    """
    file_name = str(article.pmid) + '.txt'
    path = Path(os.path.dirname(os.path.abspath(__file__))) / 'cache'

    if os.path.exists(path / 'entities' / file_name) and os.path.exists(path / 'relations' / file_name):
        print("Loading cached relations and entities")
        with open(path / 'entities' / file_name, 'rb') as f:
            entities = pickle.load(f)
        with open(path / 'relations' / file_name, 'rb') as f:
            relations = pickle.load(f)
        return entities, relations

    text = ''
    if source == 'abstract':
        text = article.abstract
    elif source == 'full_text':
        text = article.full_text
    else:
        raise ValueError("Invalid source: {}".format(source))

    # TODO: Delete this pls
    # truncate the text if it is too long
    # if len(text) > 512:
    #    text = text[:512]

    entities = extract_entities(article, source)
    span_sentences = tokenize_into_sentences(article, source)

    # divide entities in gene and disease entities
    gene_entities = []
    disease_entities = []
    for entity in entities:
        if entity.type == "disease":
            disease_entities.append(entity)
        elif entity.type == "gene":
            gene_entities.append(entity)
    
    relations = []
    for gene_entity in gene_entities:
        for disease_entity in disease_entities:

            #find the sentence that contains the gene and disease
            sentence_index_gene = find_entity(gene_entity, span_sentences)
            sentence_index_disease = find_entity(disease_entity, span_sentences)

            masked_text = ''
            if gene_entity.span_begin < disease_entity.span_begin:
                masked_text = text[span_sentences[sentence_index_gene][0]:gene_entity.span_begin] + "@GENE$" + text[gene_entity.span_end:disease_entity.span_begin] + "@DISEASE$" + text[disease_entity.span_end:span_sentences[sentence_index_disease][1]]
            else:
                masked_text = text[span_sentences[sentence_index_disease][0]:disease_entity.span_begin] + "@DISEASE$" + text[disease_entity.span_end:gene_entity.span_begin] + "@GENE$" + text[gene_entity.span_end:span_sentences[sentence_index_gene][1]]
            try:
                out = rel_pipe(masked_text)[0]

                if out['label'] == 'LABEL_1':
                    # TODO: Add the score as the probability
                    relations.append((gene_entity, disease_entity))
            except Exception as e:
                print(e)
                continue
    
    # save entities and relations to file with pickle
    with open(path / 'entities' / file_name, 'wb') as f:
        pickle.dump(entities, f)
    with open(path / 'relations' / file_name, 'wb') as f:
        pickle.dump(relations, f)
    
    return entities, relations

def query_plain(text, url="http://bern2.korea.ac.kr/plain"):
    """
    Query the BERN2 server for plain text.
    """
    # Limit the text at 5000 characters
    if len(text) > 5000:
        text = text[:5000]
    
    result = requests.post(url, json={'text': text})

    if result.status_code != 200:
        # Sleep for 10 seconds and try again
        print("Error: {}".format(result.status_code))
        raise Exception("Error: {}".format(result.status_code))
    
    return result.json()

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