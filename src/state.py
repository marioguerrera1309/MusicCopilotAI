from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langchain_core.messages import AnyMessage #AnyMessage, nativo di LangChain, supporta i vari SystemMessage, HumanMessage e ToolMessage
from langgraph.graph.message import add_messages


# Questo file definisce la struttura dati condivisa su cui tutti i nodi leggono e scrivono.
# Ai vari nodi, e funzioni utilizzate dai nodi, viene passato un'istanza di questa classe
class AgentState(TypedDict):
    proposed_posts: List[Dict[str, Any]]  # Sequenza di post pianificati
    current_post_index: int               # Indice del post corrente da elaborare
    current_target_topic: Dict[str, Any]  # Topic del post corrente su cui scrivere il post
    
    context_knowledge_graph: Dict[str, Any] #contenuto recuperato dal Knowledge_graph
    recovered_documents_RAG: List[Dict[str, Any]] #documenti recuperati dal ChromaDB
    
    # Per implementare il pattern ReAct di LangGraph, è presente una lista di messaggi 
    # che accumula i Thought dell'LLM e le Observation dei tool
    messages: Annotated[List[AnyMessage], add_messages] #add_message, funzione nativa di LangGraph, fa si che ogni volta che c'è un nuovo messaggio, la lista non venga sovrascritta
    research_status: str                  # può essere "COMPLETE" o "IN_PROGRESS"
    
    generated_draft: str                  # Bozza di testo dell'articolo
    feedback: str                         # può essere "approved", "modified", "rejected"
    notes: Optional[str]                  # Note di modifica dall'utente, nel caso in cui selezioni "modified". Queste verranno poi acquisite dal writer 

    approved_posts: List[Dict[str, Any]]   # Post approvati e consolidati
    publication_schedule: List[Dict[str, Any]]  # Piano di pubblicazione su database
    published_posts: List[Dict[str, Any]]   # Post pubblicati
    posts_to_publish: List[Dict[str, Any]]  # Post ancora in coda di pubblicazione
    publication_summary: str               # Riepilogo leggibile della simulazione