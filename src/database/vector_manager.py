import os
import sys
# Importiamo l'integrazione ufficiale di LangChain per ChromaDB
from langchain_chroma import Chroma
# Importiamo il modulo di LangChain per interfacciarsi con i modelli di embedding di Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Allineamento dei path, per ritornare alla root principale. 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import get_google_api_key #per recuperare le chiavi GOOGLE_API_KEY_1 e GOOGLE_API_KEY_2

class VectorRAGManager:
    #nell'__init__ impostiamo la cartella di default dove salvare i dati 
    def __init__(self, persist_directory="./data/chroma_db"):
        """Inizializza il Vector DB Chroma usando gli Embedding ufficiali di Google."""
        self.persist_directory = persist_directory
        
        # Impostiamo il modello di embedding usando il modello standard di Google Gemini
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2", 
            google_api_key=get_google_api_key()
        )
        
        # Inizializziamo o carichiamo il database vettoriale locale, che poi usiamo per salvare i testi
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="local_knowledge_rag" #definisce il nome della tabella interna nel DB
        )

    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        """
        Aggiunge una lista di testi al Vector DB, generando automaticamente 
        i vettori (embedding) e salvandoli su disco.
        """
        if not texts:
            return
        
        print(f"Generazione embedding e indicizzazione di {len(texts)} documenti...")
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        print("Salvataggio completato su disco.")

    def search_relevant_docs(self, query: str, k: int = 3) -> list[dict]:
        """
        Esegue una ricerca semantica (Similarity Search).
        Ritorna i 'k' documenti più vicini a livello di significato rispetto alla query.
        """
        print(f"Ricerca semantica per query: '{query}'")
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_docs = [] #lista di dizionari
        # Cicla sui risultati restituiti da ChromaDB (ogni risultato è una tupla composta dall'oggetto Document e dallo Score)
        for doc, score in results:
            formatted_docs.append({
                "content": doc.page_content, #viee aggiunto il contenuto testuale vero e proprio
                "metadata": doc.metadata,   #vengono aggiunti i metadati del dizionario "metadata" relativo al chunck, in cui è contenuto il numero di pagina e il nome del file
                "score": float(score)
            })
        return formatted_docs