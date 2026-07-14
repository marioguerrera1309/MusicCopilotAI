import os
import sys
import re
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage

# Allineamento dei path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database.vector_manager import VectorRAGManager
from src.database.kg_manager import EditorialKGManager

def _extract_keywords_for_expansion(text_list: list) -> str:
    """
    Utility interna K-RAG: prende una lista di testi strutturati (claim),
    rimuove le parole inutili e restituisce solo i token tecnici unici 
    da usare per l'espansione della query.
    """
    if not text_list:
        return ""
    
    stop_words = {
        "la", "il", "lo", "i", "gli", "le", "un", "una", "uno", "del", "dell", 
        "della", "dei", "degli", "delle", "sullo", "sulla", "sulle","sugli", "con", "per", "tra", "fra", "dopo",
        "guida", "pratica", "tecniche", "avanzate", "manuale", "articolo", "post",
        "le", "ricerche", "condotte", "non", "hanno", "restituito", "informazioni"
    }

    # Unisce tutti i claim testuali storici in un'unica stringa forzando il minuscolo
    combined_text = " ".join(text_list).lower()
    # Isola solo i token alfanumerici (parole e numeri), eliminando punteggiatura e caratteri speciali
    raw_words = re.findall(r'[a-zA-Z0-9àèìòù]+', combined_text)
    
    # Seleziona parole chiave uniche, non stop-word e con lunghezza significativa
    clean_words = []
    for w in raw_words:
        if w not in stop_words and len(w) >= 4 and w not in clean_words:
            clean_words.append(w)
            
    # Restituiamo le prime 4 parole chiave più dense come "vettore di espansione"
    return " ".join(clean_words[:4])

@tool
def rag_retrieval_tool(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Interroga il database vettoriale locale (ChromaDB) implementando il pattern K-RAG:
    perfeziona ed espande la query originale sfruttando la conoscenza strutturata
    del Knowledge Graph (Neo4j) per garantire grounding e coerenza.
    """
    print(f"-> [K-RAG TOOL] Avvio recupero aumentato da Grafo per la query: '{query}'")
    
    
    # Istanziamo l'EditorialKGManager 
    kg_manager = EditorialKGManager()
    expanded_query = query #se viene trovato un topic correlato, si espande la query, altirmenti rimarrà questa stessa
    kg_enrichment_applied = False #flag che indica che è stata espansa la query
    
    try:
        # Interroghiamo Neo4j
        kg_context = kg_manager.get_context_for_topic(query)
        
        if kg_context.get("exists") and kg_context.get("previous_claims"):
            # Invochiamo la funzione di estrazione delle 4 parole chiave di tutti i claim contenuti nella lista del dizionario
            kg_keywords = _extract_keywords_for_expansion(kg_context["previous_claims"])
            
            if kg_keywords:
                #aggiorna la variabile expanded query aggiungendo pure le parole chiave ritornate dalla funzione precedente
                expanded_query = f"{query} {kg_keywords}".strip()
                kg_enrichment_applied = True #aggiornamento del flag
                print(f"-> [K-RAG] Query espansa con successo tramite Keyword strutturate dal KG: '{expanded_query}'")
        
        if not kg_enrichment_applied:
            print("-> [K-RAG] Il topic è un GAP nel grafo. Come da requisiti, la query RAG userà il topic nativo.")
            
    except Exception as e:
        print(f"-> [WARNING] Impossibile espandere la query con il KG: {e}")
    finally:
        kg_manager.close()
    
    #Retrieval dei documenti da ChromaDb
    try:
        rag_manager = VectorRAGManager()
        relevant_docs = rag_manager.search_relevant_docs(expanded_query, k=2)
        
        # Se il database vettoriale restituisce una lista vuota (nessun chunk trovato)
        if not relevant_docs:
            message_content = "Nessun documento semanticamente rilevante trovato nell'archivio locale RAG."
            # Ritorna un comando di aggiornamento vuoto per non bloccare il ciclo dell'agente
            return Command(
                update={
                    "recovered_documents_RAG": [],
                    "messages": [ToolMessage(content=message_content, name="rag_retrieval_tool", tool_call_id=tool_call_id)]
                },
            )
        #Se vengono restituiti documenti rilevanti (insieme ai metadati)
        #Costruisce l'intestazione dell'Observation testuale per il Researcher
        output_blocks = [
            f"[OBSERVATION - RETRIEVAL INTEGRATO K-RAG]",
            f"-> QUERY ORIGINALE DELL'AGENTE: '{query}'",
            f"-> QUERY PERFEZIONATA DAL KNOWLEDGE GRAPH: '{expanded_query}'",
            f"=== DOCUMENTI STRUTTURATI/NON STRUTTURATI RECUPERATI VIA CHROMADB ==="
        ]
        state_documents = [] #inizializziamo la lista con cui aggironiamo lo stato
        
        # Cicla sui chunk estratti da ChromaDB (formattati come dizionari dal vector_manager)
        for i, doc in enumerate(relevant_docs, 1):  #partiamo con i=1 anzichè i=0
            src = doc["metadata"].get("source", "Archivio Sconosciuto") # Estrae il titolo dai metadati impostando un valore di fallback protettivo
            page = doc["metadata"].get("page", "N/A") # Estrae la pagina originale
            
            # Appende il blocco informativo testuale per la chat del Researcher (mostrando anche lo Score)
            output_blocks.append(
                f"[Documento Locale #{i} - Fonte: {src} (Pag. {page})] (Distanza Vettoriale: {doc['score']:.4f})\n"
                f"Contenuto: {doc['content']}\n"
                f"---"
            )
            
            #Aggiorniamo la lista di dizionari state_document con un nuovo dizionario contenente il contenuto estratto, fonte e pagina e score.
            #E' diverso rispetto al formatted_docs restituito dal vector_manager perchè i metadati (titolo e pagina) vengono estratti 
            state_documents.append({
                "content": doc["content"],
                "source": f"{src} (Pag. {page})",
                "score": doc["score"]
            })
            
        # Unisce tutti i blocchi testuali inserendo un doppio carattere di "a capo" (\n\n)
        final_text_observation = "\n\n".join(output_blocks)
        
        return Command(
            update={
                # Aggiorna la chiave dell'AgentState con l'elenco strutturato dei chunk (per il grounding del Writer)
                "recovered_documents_RAG": state_documents,
                # Invia il report discorsivo completo al Researcher per permettergli di proseguire il ReAct
                "messages": [ToolMessage(content=final_text_observation, name="rag_retrieval_tool", tool_call_id=tool_call_id)]
            },
        )
        
    except Exception as e:
        error_msg = f"Errore durante il recupero dei documenti dal Vector DB locale: {str(e)}"
        return Command(
            update={
                "recovered_documents_RAG": [],
                "messages": [ToolMessage(content=error_msg, name="rag_retrieval_tool", tool_call_id=tool_call_id)]
            },
        )