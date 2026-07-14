import os
import sys
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config.prompts import PLANNER_SYSTEM_PROMPT

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.state import AgentState
from src.database.kg_manager import EditorialKGManager
from config.config import get_google_api_key

#Se l'applicazione dovesse crashare a causa di un errore 503 ai server di google, usiamo una pianificazione d'emergenza. 
# Di volta in volta vanno aggiornati i topic, altrimenti vengono generati post già presenti 

def _build_fallback_plan() -> list[dict]:
    """Genera una sequenza editoriale autonoma di emergenza incentrata sul dominio musicale."""
    print("-> [PLANNER FALLBACK] Generazione di un piano editoriale predefinito per il dominio della musica.")
    return [
        {
            "topic": "La rivoluzione del Grunge e l'evoluzione dell'Alternative Rock",
            "type": "Domain advances",
            "justification": "Piano autonomo di ripiego per coprire le lacune storiche del rock anni '90."
        },
        {
            "topic": "Guida pratica alla regolazione dei preamplificatori in un mixer da studio",
            "type": "How to",
            "justification": "Contenuto tecnico d'archivio fondamentale per bilanciare la presenza di guide pratiche."
        },
        {
            "topic": "Analisi timbrica e recensione dei sintetizzatori analogici monofonici",
            "type": "Product review",
            "justification": "Recensione mirata per diversificare l'offerta editoriale del blog."
        }
    ]


def plan_topics_node(state: AgentState) -> dict:
    """
    Nodo del Planner Autonomo: Interroga lo storico di Neo4j per identificare
    le aree coperte, pianifica una sequenza strategica di post diversificati
    e giustifica le proprie scelte editoriali.
    """
    print("\nAnalisi gap su Neo4j e Pianificazione ---")
    
    #Istanziamo EditorialKGManager del kg_manager.py per il recupero dello storico del database Neo4J
    kg_manager = EditorialKGManager()
    try:
        #Invochiamo la funzione dell'EditorialKGManager, che restituisce una lista di dizionari di topic già trattati
        covered_topics = kg_manager.get_covered_topics_and_recency()
    except Exception as e:
        print(f"ERRROE {e}. Procedo con storico vuoto.")
        covered_topics = []
    finally:
        kg_manager.close() # Rilascia immediatamente la connessione
    
    print(covered_topics)
    #Configurazione l'LLM dedicato alla pianificazione editoriale
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.3, # Bassa per rimanere strutturato, ma sufficiente per diversificare
    )
    
    #Creiamo il template del prompt strutturato, suddividendo il system prompt dalle istruzioni dell'utente
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", (
            "Analizza attentamente lo storico dei topic già estratti e registrati nel nostro database a grafo:\n"
            "### STORICO ARGOMENTI TRATTATI (NEO4J) ###\n"
            "{kg_history}\n\n"
            "Sulla base del nostro dominio editoriale, identifica le lacune di copertura semantica "
            "e pianifica una sequenza strategica di esattamente 3 post completamente NUOVI.\n"
            "Assicurati che l'ordine sia giustificato e che non vi siano ridondanze.\n"
            "Rispondi ESCLUSIVAMENTE con la lista JSON strutturata, senza introduzioni o saluti."
        ))
    ])
    
    # Serializza la lista dello storico di Neo4j trasformandola in una stringa JSON formattata ed elegante
    formatted_prompt = planner_prompt.format_messages(
        kg_history=json.dumps(covered_topics, indent=2)
    )

    #L'llm restituisce una stringa. Siamo poi noi ad estrarre, tramite regex, il json da questa stringa, per poi convertirlo in una lista di dizionari
    #da cui estrarre il primo post per la produzione
    try:
        response = llm.invoke(formatted_prompt)
        # Rimuove gli spazi vuoti o i caratteri di "a capo" all'inizio e alla fine del testo ricevuto
        clean_content = response.content.strip()
        
        # Parsing robusto tramite espressioni regolari per isolare l'array JSON
        json_match = re.search(r'\[\s*\{.*\}\s*\]', clean_content, re.DOTALL)
        if json_match:
            #estrae e decodifica solo la stringa JSON pura isolata, per convertirla in una lista di dizionari
            plan = json.loads(json_match.group(0))
        else:
            #Fallback se la regex fallisce: ripulisce manualmente i tag di blocco codice markdown di Gemini (```json)
            clean_content = clean_content.replace("```json", "").replace("```", "").strip()
            # Tenta la decodifica sull'intera stringa ripulita
            plan = json.loads(clean_content)
            
        #Se l'llm ha risposto con un singolo dict, invece che una lista, lo trasforma in lista di dict contenente quel singolo dict
        if isinstance(plan, dict):
            plan = [plan]
            
        if not isinstance(plan, list) or not plan:
            raise ValueError("L'output non contiene una lista valida di pianificazione.")
            
        print(f"-> [SUCCESS] Regia Editoriale: Sequenza di {len(plan)} post generata con successo.")
    except Exception as e:
        print(f"-> [WARNING] Errore nel parsing del piano dell'LLM: {e}. Attivo il Fallback autonomo.")
        plan = _build_fallback_plan()
        
    # Estraiamo l'oggetto corrispondente al primo post della roadmap (Indice 0)
    first_post = plan[0] if plan else {"topic": "Unknown Topic", "type": "Domain advances"}
    
    #Restituiamo uno stato pulito. Inizializziamo i messaggi a vuoto 
    # per accogliere il primo ciclo ReAct del researcher senza scorie precedenti.
    return {
        "proposed_posts": plan, # Salva l'intera roadmap di 3 post generata (o di fallback)
        "current_post_index": 0,    # Inizializza il contatore dei post da realizzare posizionandolo sul primo elemento
        "current_target_topic": first_post, # Imposta il target immediato che guiderà il Researcher nel primo ciclo
        "messages": [],
        "research_status": "IN_PROGRESS",
        "approved_posts": [],
        "publication_schedule": [],
        "published_posts": [],
        "posts_to_publish": [],
        "publication_summary": ""
    }