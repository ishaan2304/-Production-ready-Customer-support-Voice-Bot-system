"""
RAG (Retrieval Augmented Generation) Module.
Uses ChromaDB as vector store and LangChain for orchestration.
Retrieves relevant knowledge base documents before generating responses.
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.config import get_config
from app.exceptions import VoiceBotError
from app.logger import get_logger

logger = get_logger(__name__)


class RAGModule:
    """
    Retrieval Augmented Generation using ChromaDB + LangChain.
    Loads company knowledge base and retrieves relevant context
    for each user query before passing to the LLM.
    """

    def __init__(self):
        self.config = get_config()
        self._vectorstore = None
        self._embeddings = None
        self._retriever = None
        self._loaded = False
        self._kb_path = Path(__file__).resolve().parent.parent / "data/knowledge_base.json"
        self._chroma_path = Path(__file__).resolve().parent.parent / "models/chroma_db"

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_openai import OpenAIEmbeddings
            from langchain.schema import Document
            import os
            from dotenv import load_dotenv
            load_dotenv()

            logger.info("Initializing RAG module with ChromaDB...")

            self._embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )

            # Load or create vector store
            self._chroma_path.mkdir(parents=True, exist_ok=True)
            chroma_dir = str(self._chroma_path)

            # Check if already indexed
            if (self._chroma_path / "chroma.sqlite3").exists():
                logger.info("Loading existing ChromaDB index...")
                self._vectorstore = Chroma(
                    persist_directory=chroma_dir,
                    embedding_function=self._embeddings,
                    collection_name="voicebot_kb",
                )
            else:
                logger.info("Building ChromaDB index from knowledge base...")
                docs = self._load_knowledge_base()
                self._vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self._embeddings,
                    persist_directory=chroma_dir,
                    collection_name="voicebot_kb",
                )
                logger.info(f"Indexed {len(docs)} documents into ChromaDB")

            self._retriever = self._vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},
            )
            self._loaded = True
            logger.info("RAG module ready")

        except ImportError as e:
            logger.error(f"RAG dependencies not installed: {e}")
            raise VoiceBotError(f"RAG dependencies missing: {e}")
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}", exc_info=True)
            raise VoiceBotError(f"RAG initialization failed: {e}")

    def _load_knowledge_base(self):
        """Load knowledge base JSON and convert to LangChain Documents."""
        from langchain.schema import Document

        with open(self._kb_path) as f:
            kb_data = json.load(f)

        documents = []
        for item in kb_data:
            doc = Document(
                page_content=item["content"],
                metadata={
                    "id": item["id"],
                    "category": item["category"],
                    "title": item["title"],
                },
            )
            documents.append(doc)
        return documents

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query text
            k: Number of documents to retrieve

        Returns:
            List of relevant document dicts
        """
        self._lazy_load()
        start = time.perf_counter()

        try:
            docs = self._retriever.invoke(query)
            elapsed = (time.perf_counter() - start) * 1000

            results = [
                {
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", ""),
                    "category": doc.metadata.get("category", ""),
                    "score": None,
                }
                for doc in docs
            ]

            logger.debug(
                f"RAG retrieved {len(results)} docs for '{query[:50]}' in {elapsed:.1f}ms"
            )
            return results

        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}", exc_info=True)
            return []

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not documents:
            return "No specific knowledge base articles found for this query."

        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(
                f"[Article {i}: {doc['title']}]\n{doc['content']}"
            )
        return "\n\n".join(parts)

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add new documents to the knowledge base."""
        self._lazy_load()
        from langchain.schema import Document

        docs = [
            Document(
                page_content=d["content"],
                metadata={"title": d.get("title", ""), "category": d.get("category", "")},
            )
            for d in documents
        ]
        self._vectorstore.add_documents(docs)
        logger.info(f"Added {len(docs)} documents to ChromaDB")
