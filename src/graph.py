from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver #per salvare lo stato tra i vari step
from langgraph.prebuilt import ToolNode #componente predefinito di langgraph per gestire l'esecuzione automatica dei tool
from langgraph.types import Command 
import datetime
import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) #effettua i vari import partendo dalla root principale del progetto (utile se per esempio si dovesse eseguire il progetto da terminale non nella root principale)
from src.database.kg_manager import EditorialKGManager
from src.state import AgentState #stato condiviso dai vari nodi
from src.agents.planner import plan_topics_node
from src.agents.researcher import react_research_node
from src.agents.writer import write_post_node
from src.database.kg_manager import update_kg_node
from src.tools.search_tool import web_search_tool
from src.tools.rag_tool import rag_retrieval_tool
from src.tools.kg_tool import query_knowledge_graph_tool
from src.tools.credibility_tool import source_credibility_tool
from src.tools.web_content_moderator_tool import web_content_moderator_tool

tools_list = [web_search_tool, rag_retrieval_tool, query_knowledge_graph_tool, source_credibility_tool, web_content_moderator_tool]
tool_node = ToolNode(tools_list) #il ToolNode intercetta le richieste dell'utente ed esegue i tool

# E' la funzione corrispondente al nodo human review. 
# Funge da punto di aggancio passivo per l'interruzione. Non modifica lo stato.
def human_review_node(state: AgentState):
    print("-> [HUMAN REVIEW] Il sistema è in attesa di feedback sulla GUI...")
    return {}

# Viene eseguita dal nodo advance_post
# Incrementa l'indice per scorrere la lista (di dizionari) dei post pianificati dal planner. 
# Resetta alcuni campi dello  stato relativi al post precedente, ma è più un controllo di sicurezza, visto che già viene effettuato in update:knowledge_graph
def advance_to_next_post_node(state: AgentState):
    idx = state.get("current_post_index", 0) #recupera l'indice attuale del post. Se, non esiste, mette 0
    new_idx = idx + 1
    posts = state.get("proposed_posts", []) #recupera i post pianificati dal planner
    
    updates = {"current_post_index": new_idx} #Prepara il dizionario in cui, tra le tante modifiche, ci sarà l'aggiornamento dell'indice 
    if new_idx < len(posts): #se la disuguaglianza è soddisfatta vuol dire che ancora c' uno/due post da realizzare
        updates["current_target_topic"] = posts[new_idx] # Imposta il nuovo topic di riferimento per il prossimo post, 
        # Resettiamo i messaggi e lo stato di ricerca per il prossimo articolo
        updates["messages"] = [] 
        updates["research_status"] = "IN_PROGRESS"
        updates["feedback"] = ""
    return updates

def route_after_rejected(state: AgentState):
    return{
        "messages": [],
        "notes": "",
        "recovered_documents_RAG": [],
        "context_knowledge_graph": {},
        "generated_draft": "",
        "research_status": "IN_PROGRESS",
        "feedback": "",
        # Vengono lasciati inalterati i contatori e lo stato globale del blog
        "approved_posts": state.get("approved_posts", []),
        "publication_schedule": state.get("publication_schedule", []),
        "published_posts": state.get("published_posts", []),
        "posts_to_publish": state.get("posts_to_publish", []),
        "publication_summary": state.get("publication_summary", "")
    }


#Viene eseguita dal nodo simulate_publication. Se nel publication_state.json 
# il campo 'ultimo controllo' ha una data di ieri o più giorni, la funzione pubblica sul database (Neo4j) 
# il post più antico del publication_state.json 
def simulate_blog_publication_node(state: AgentState):
    """
    Funzione Finale del Workflow:
    Funge da 'Guardiano del Tempo'. Controlla il JSON e, se è passato almeno 1 giorno,
    chiama add_approved_post per pubblicare il post più vecchio su Neo4j,
    rimuovendolo stabilmente dal file e aggiornando il timestamp.
    """
    json_path = "publication_state.json"
    oggi = datetime.date.today()
    
    if not os.path.exists(json_path):
        print("Nessun file JSON trovato. Coda vuota, nulla da pubblicare.")
        return {"publication_summary": "Coda vuota."}
        
    with open(json_path, "r", encoding="utf-8") as f:
        buffer_data = json.load(f)
        
    if "queue" not in buffer_data or not buffer_data["queue"]: #verifica che la chiave queue esista e che non sia vuota
        print("Nessun post in attesa nel JSON.")
        return {"publication_summary": "Nessun post in coda nel JSON."}

    
    data_passata = (oggi - datetime.timedelta(days=10)).isoformat() #per evitare che il sistema si blocchi, se non c'è un ultimo controllo, lo si imposta a dieci giorni fa
    ultimo_controllo_str = buffer_data.get("ultimo_controllo", data_passata)
    ultimo_controllo = datetime.date.fromisoformat(ultimo_controllo_str)
    
    giorni_passati = (oggi - ultimo_controllo).days
    print(f"Ultimo controllo: {ultimo_controllo} | Oggi: {oggi} | Giorni passati: {giorni_passati}/1")

    published_posts = []
    successo_pubblicazione = False

    #Controlla se è passato almeno 1 giorno
    if giorni_passati >= 1:
        print("Controllo superato. Estraggo il post più antico.")
        
        
        buffer_data["queue"].sort(key=lambda x: x["saved_at"]) #ordina la cosa in base al timestamp di salvataggio
    
        post_da_pubblicare = buffer_data["queue"].pop(0) #Estrae il post più vecchio e lo rimuove, assegnandolo ad una nuova variabile usata per caricarlo sul database
        
        #Per pubblicare il post su Neo4j, si utilizza la funzione add_approved_post del kg_manager.py
        kg_manager = EditorialKGManager()
        try:
            kg_manager.add_approved_post(
                post_id=post_da_pubblicare["post_id"],
                title=post_da_pubblicare["title"],
                content=post_da_pubblicare["content"],
                target_topic=post_da_pubblicare["topic"],
                claims=post_da_pubblicare["claims"],
                sources=post_da_pubblicare["sources"]
            )
            print(f"Post '{post_da_pubblicare['title']}' publicato correttamente!")
            
            successo_pubblicazione = True
            published_posts.append(post_da_pubblicare) #aggiunge alla lista il post appena pubblicato
            
        except Exception as e:
            print(f"-> [ERRORE CRITICO NEO4J] add_approved_post fallito: {e}")
            # Failsafe: se il database fallisce, si reinserisce il post in cima alla coda nativa
            buffer_data["queue"].insert(0, post_da_pubblicare)
        finally:
            kg_manager.close()
            
        
        if successo_pubblicazione: #se il flag è stato settato in quanto la pubblicazione è andata a buon fine, si aggiorna la data
            buffer_data["ultimo_controllo"] = oggi.isoformat()
            print(f"'ultimo_controllo' aggiornato a oggi: {oggi.isoformat()}")
            
    else:
        print(f" Non è ancora passato 1 giorno. Coda congelata.")

    #riscrive il json aggiornato all'interno del file, con il nuovo 'ultimo controllo' e i post rimanenti
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(buffer_data, f, indent=2, ensure_ascii=False)
    
    rimanenti_in_coda = len(buffer_data["queue"])
    summary = f"Rilasciato il post su Neo4j. In coda nel JSON rimangono {rimanenti_in_coda} post."
    print(f"FINE {summary}\n")

    past_published = state.get("published_posts") or []

    #Alla fine, aggiorna lo stato del sistema
    return {
        "publication_schedule": buffer_data["queue"],
        "published_posts": past_published + published_posts,
        "posts_to_publish": buffer_data["queue"],
        "publication_summary": summary,
        "proposed_posts": [],
        "current_post_index": 0,
        "current_target_topic": {}
    }

workflow = StateGraph(AgentState)

# Aggiunta Nodi
workflow.add_node("planner", plan_topics_node)
workflow.add_node("researcher", react_research_node)
workflow.add_node("tools", tool_node) # Nodo dei tool per il ReAct
workflow.add_node("writer", write_post_node)
workflow.add_node("human_review", human_review_node) 
workflow.add_node("route_after_rejected", route_after_rejected) 
workflow.add_node("update_knowledge_graph", update_kg_node)
workflow.add_node("advance_post", advance_to_next_post_node)
workflow.add_node("simulate_publication", simulate_blog_publication_node)

# Il flusso parte dal planner, e poi si collega al researcher
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")

#viene eseguita dopo il nodo researcher
def route_after_research(state: AgentState):
    # Se il researcher ha deciso di aver finito, e lo si vede dal campo research_status, allora restituisce una stringa "go_to_writer" indirizzando il flusso verso il writer
    status = state.get("research_status", "")
    if status == "COMPLETE":
        return "go_to_writer"
    
    # Altrimenti, se l'LLM ha generato una richiesta di chiamata tool nell'ultimo messaggio, andiamo al ToolNode
    messages = state.get("messages", [])
    if messages and messages[-1].tool_calls: #se l'ultimo messaggio esiste, ed è una chiamata ad un tool, restituisce una stringa "execute tools"
        return "execute_tools"
        
    return "go_to_writer"


workflow.add_conditional_edges(
    "researcher",
    route_after_research,
    {
        "execute_tools": "tools",   # Va a eseguire il tool fisico
        "go_to_writer": "writer"    # Finisce la ricerca e passa alla scrittura
    }
)
# Il ToolNode, dopo aver eseguito l'azione, restituisce SEMPRE il controllo al researcher (l'observation è seguita dal thought)
workflow.add_edge("tools", "researcher") 

#dopo il writer, il sistema rimane in attesa di un feedback umano
workflow.add_edge("writer", "human_review") 

#funzione eseguita dopo il nodo juman review
def route_after_human_review(state: AgentState):
    #si recupera il campo feedback dallo state, e si rimuovono eventuali spazi e si trasforma tutto in minuscolo
    feedback = str(state.get("feedback", "")).strip().lower()
    print(f"Feedback rilevato: '{feedback}' ---")
    
    #in base al feedback, si ritornano diverse stringhe che attivano diversi nodi
    if feedback == "approved":
        return "update_kg"
    elif feedback == "rejected":
        return "researcher_draft"
    elif feedback == "modified":
        return "rewrite_draft"
    else:
        return "rewrite_draft"



workflow.add_conditional_edges(
    "human_review", 
    route_after_human_review,
    {
        "update_kg": "update_knowledge_graph",
        "researcher_draft": "route_after_rejected",
        "rewrite_draft": "writer"            # Se modificato riscrive direttamente
    }
)
workflow.add_edge("route_after_rejected","researcher")

workflow.add_edge("update_knowledge_graph", "advance_post")

#funzione eseguita dopo che il nodo advance_post ha incrementato l'indice e aggiornato lo state 
def route_for_next_post(state: AgentState):
    posts = state.get("proposed_posts", [])
    idx = state.get("current_post_index", 0)
    
    print(f"Verifico indice post: {idx}/{len(posts)} ---")
    if idx < len(posts):
        return "prossimo_post"
    else:
        return "fine_sequenza"

workflow.add_conditional_edges(
    "advance_post",
    route_for_next_post,
    {
        "prossimo_post": "researcher", # Ricomincia il ciclo ReAct sul nuovo topic
        "fine_sequenza": "simulate_publication"           # Conclude con la pubblicazione nel database
    }
)

workflow.add_edge("simulate_publication", END)
# Rileva se il file sta girando su LangGraph  o localmente tramite il main.py (con la GUI Tkinter)
is_platform = (os.getenv("LANGGRAPH_API") is not None or os.getenv("LANGGRAPH_PORT") is not None or "langgraph" in sys.argv[0])
if is_platform:
    print("Compilazione ottimizzata per LangGraph Cloud/Dev CLI.")
    # In ambiente cloud non usiamo MemorySaver locale, lasciamo che sia la piattaforma a gestire la persistenza
    compiled_graph = workflow.compile(interrupt_before=["human_review"])
else:
    print("Compilazione con MemorySaver locale per Tkinter.")
    # In ambiente locale istanziamo MemorySaver in RAM, necessario per consentire alla GUI Tkinter di fare l'interruzione
    checkpointer = MemorySaver()
    # Compila il grafo dicendogli di congelarsi tassativamente PRIMA di entrare nel nodo 'human_review'
    compiled_graph = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])