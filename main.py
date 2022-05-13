

from pubmad import *


articles: List[Article] = download_articles('diabetes', start_year=2019, end_year=2020, max_articles=100, sort_by='relevance')