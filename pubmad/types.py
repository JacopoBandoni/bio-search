from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class Article:
    title: str
    abstract: Optional[str]
    full_text: Optional[str]
    pmid: str
    publication_data: datetime


class Entity:
    '''
    Entity defines a gene or a disease in a text.

        Parameters

            mesh_id (List[str]): List of Unique identifier for the entity, e.g. MESH:D001234. (You can use the MESH ID to look up the name of the entity in the MESH database.)
            mention (str): The string that is associated with the entity in the text (e.g. "BRCA1").
            type (str): The type of the entity. Can be 'gene' or 'disease'.
            prob (float): The probability of the entity, it indicates the confidence of the model prediction.
            span_begin (int): The begin of the span of the entity.
            span_end (int): The end of the span of the entity.
            article (Article): The article the entity is associated with.
            source (str): The source of the entity within the article. Can be 'abstract' or 'full_text'. Defaults to 'abstract'.
            pmid (str): The PubMed ID of the article.
    '''
    def __init__(self, mesh_id: List[str], mention: str, type: str, prob: float, span_begin: int, span_end: int, article: Article = None, source: str = 'abstract', pmid: str = ''):
        self.mesh_id = mesh_id
        self.mention = mention
        self.type = type
        self.prob = prob
        self.span_begin = span_begin
        self.span_end = span_end
        self.article = article
        self.source = source
        self.pmid = pmid