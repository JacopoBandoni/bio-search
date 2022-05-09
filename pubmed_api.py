
from turtle import down
from Bio import Entrez, Medline
from typing import List, Tuple

handle = Entrez.esearch(db="pubmed", term='diabetes', retmax=10, sort='relevance')
records = Entrez.read(handle)
print('\n'.join(records['IdList']))

def download_articles(title: str, start_year: int, end_year: int, max_results: int = 100, author: str = '', type_research :str = 'relevance'):
    """
    Download articles from PubMed using BioPython library.
    Args:
        query (str): The query to search for.
        start_year (int): The start year to search for.
        end_year (int): The end year to search for.
        max_results (int): The maximum number of results to return.
        author (str): The author to search for, leave empty to search for all authors.
    Returns:
        List[Article] A list of articles.
    """

    Entrez.email = 'pubmadbiosearch@gmail.com'

    if author != "":
        query = '(' + title + ') AND (' + author + '[Author])'
    else:
        query = title 

    handle = Entrez.esearch(db='pubmed', 
                            sort=type_research, 
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

    print([r['PMID'] for r in results])

download_articles('diabetes',start_year=1800, end_year=2022,max_results=1)