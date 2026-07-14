import os
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
# Importa l'integrazione ufficiale di LangChain per Tavily (il motore di ricerca web ottimizzato per agenti AI)
from langchain_community.tools.tavily_search import TavilySearchResults

@tool("web_search_tool")
def web_search_tool(query: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Cerca informazioni aggiornate su Internet relative a eventi recenti, 
    news di dominio, concerti imminenti in Sicilia o dettagli specifici non presenti nel DB locale.
    Input deve essere una stringa di query chiara.
    """
    #verifichiamo se è presente la variabile d'ambiente
    if not os.getenv("TAVILY_API_KEY"):
        print(" [WARNING] Chiave TAVILY_API_KEY non trovata nelle variabili d'ambiente. Eseguo Mock.")
        #Inizializziamo una stringa di errore, con cui COmmand aggirona lo stato
        error_msg=f"[MOCK SEARCH] Risultato di ricerca web simulato per: '{query}'. (Configura TAVILY_API_KEY nel file .env per attivarlo realmente)"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, name="web_search_tool", tool_call_id=tool_call_id)]
            },
        )
    try:
        # Impostiamo max_results=3 per non saturare la finestra di contesto di Gemini con testo inutile
        search = TavilySearchResults(max_results=3)
        results = search.invoke({"query": query}) #viene restituita una lista di dizionari
        print(f"-> Risultati della ricerca web per '{query}': {results}")
        
        # Inizializziamo l'array di stringhe per formattare i risultati per l'LLM del Researcher
        formatted_results = ["[OBSERVATION]=== RISULTATI RICERCA WEB ESTERNA (TAVILY) ===", 
                             f"Evidenze pubbliche trovate in tempo reale su internet per la query '{query}':\n"]
        for res in results:
            # Estrae l'URL della pagina web sorgente impostando una stringa di fallback se mancante
            url = res.get('url', 'URL non disponibile')
            # Estrae il frammento di testo (snippet) catturato da Tavily sul sito web
            content = res.get('content', 'Contenuto vuoto')
            formatted_results.append(f"Fonte (URL): {url}\nContenuto: {content}\n---")
            
        # Unisce tutte le righe dell'array inserendo un carattere di "a capo" (\n)
        final_text_observation = "\n".join(formatted_results)
    
        return Command(
            update={
                "messages": [ToolMessage(content=final_text_observation, name="web_search_tool", tool_call_id=tool_call_id)]
            }
        )
    except Exception as e:
        error_msg=f"Errore durante la ricerca web: {str(e)}"
        return Command(
            update={
                "messages": [ToolMessage(content=error_msg, name="web_search_tool", tool_call_id=tool_call_id)]
            },
        )

search_tool = web_search_tool #se ci dovesse essere un errore di importazione nel graph.py