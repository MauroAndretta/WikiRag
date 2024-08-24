"""
Contani the main class for the WikiRag 
"""
import os
import json
from operator import itemgetter

# custom imports
from wiki_rag.prompts import ANSWER_QUESTION_TEMPLATE_IT, ANSWER_QUESTION_TEMPLATE_EN

# langchain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import (    
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# web search
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# qdrant
from qdrant_client import QdrantClient

MODELS_CONTEXT_WINDOWS = {
    "llama3.1": 2000,
}

class WikiRag():
    """
    A class used to allow the users to make a conversation leveraging as KB the wikipedia articles.
    """

    def __init__(
            self, 
            qdrant_url: str, 
            qdrant_collection_name: str,
            expand_context: bool = True,
            verbose: bool = False):
        """
        Constructor of the class

        Args:
        qdrant_url (str): the url of the qdrant server
        qdrant_collection_name (str): the name of the collection in the qdrant server
        verbose (bool): if True, the class will print all the logs
        expand_context (bool): if True, the class will search on the web to expand the context
        """
        # Instantiate class attributes
        self.verbose = verbose
        self.expand_context = expand_context

        self.chat_ollama = ChatOllama(
            model="llama3.1",
            temperature=0,
        )

        self.huggingface_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        qdrant_client = QdrantClient(url=qdrant_url)

        # Check the qudrant collection exists
        try:
            if qdrant_client.collection_exists(qdrant_collection_name):
                collection_status = qdrant_client.get_collection(qdrant_collection_name)
                if not collection_status.status in ["green"]:
                    raise Exception(f"Collection {qdrant_collection_name} is not in a good status: {collection_status.status}")
            else:
                raise Exception(f"Collection {qdrant_collection_name} does not exist")
            
        except Exception as e:
            raise(f"Error: {e}")
        

        self.vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=qdrant_collection_name,
            embedding=self.huggingface_embeddings,
        )

        self.retriver = self.vector_store.as_retriever(
            search_kwargs={"k": 4,
                           "score_threshold": 0.5}
        )

    def get_model_name(self) -> str:
        """
        Method to get the model name
        """
        return self.chat_ollama.model

    def web_context_expansion(self, query: str) -> str:
        """
        Method to search infromation on the web to expand the context, 
        hopefully getting a better answer

        Args:
        query (str): the query to search
        """
        # check if the slef.expand_context is True
        if not self.expand_context:
            return ""
        else:
            wrapper = DuckDuckGoSearchAPIWrapper(region="it-it")

            search = DuckDuckGoSearchRun(api_wrapper=wrapper)

            # Run the search
            return search.invoke(query)
    
    
    def build_chain(self) -> Runnable:
        """
        Method to build the chain of the conversation
        """
        token_limit = 2000

        # Create the chain
        return (
            # Chain Goal: retrive k documents from the retriever
            # keys= ["query"]
            RunnableParallel(
                # retrive the web_context from the web
                web_context = (
                    RunnableLambda(lambda x: self.web_context_expansion(x["query"]))
                ),
                # retrive the context
                context = (
                    itemgetter("query")
                    | self.retriver
                ),
                query = itemgetter("query")
            )
            # Chain Goal: answer the question
            | PromptTemplate.from_template(ANSWER_QUESTION_TEMPLATE_IT)
            | self.chat_ollama
            | StrOutputParser()
    )

    def invoke(self, query: str) -> str:
        """
        Method to invoke the conversation

        Args:
        query (str): the query to ask to the model
        """
        # Run the chain
        return self.build_chain().invoke({"query": query})

