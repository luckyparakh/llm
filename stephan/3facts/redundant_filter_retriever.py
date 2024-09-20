from typing import Any, Dict, List
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document

# every time main.py runs it will load same data as embedding in Chroma DB & will cause duplicity in DB. Duplicity is good in some way but it should be controller
# Lang chain doesn't provide chain to apply filter to remove duplicate records. Hence need to create custom retriever.

# For custom retriever create a Class which has BaseRetriever as parent class & implement to below functions
class RedundantFilterRetriever(BaseRetriever):
    
    # Create object of embedding & chroma which can be passed while object creation
    embeddings: Embeddings
    chroma: Chroma
    def get_relevant_documents(self, query):
        emb=self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
            )
        return []
    async def aget_relevant_documents(self, query):
        return []