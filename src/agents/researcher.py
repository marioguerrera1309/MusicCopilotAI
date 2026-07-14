import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.prompts import RESEARCHER_SYSTEM_PROMPT
from src.state import AgentState

# Importiamo i tool come oggetti di LangChain (devono essere definiti con il decoratore @tool)
from src.tools.search_tool import web_search_tool  
from src.tools.rag_tool import rag_retrieval_tool  
from src.tools.kg_tool import query_knowledge_graph_tool  
from src.tools.credibility_tool import source_credibility_tool
from src.tools.web_content_moderator_tool import web_content_moderator_tool
from config.config import get_google_api_key

def react_research_node(state: AgentState) -> dict:
    """
    Nodo del Researcher.
    Riceve lo stato, invoca l'LLM vincolato ai tool e delega l'esecuzione a LangGraph.
    """
    print("\nCiclo di ragionamento ed esplorazione (ReAct) ---")
    
    # Recuperimao l'elenco storico dei messaggi accumulati fino ad ora nello stato corrente
    messages = state.get("messages", [])

    #Contiamo quanti messaggi emessi dall'AI contengono una richiesta di esecuzione di un tool
    num_tool_calls = len([
        msg for msg in messages 
        if isinstance(msg, AIMessage) and msg.tool_calls
    ])

    # Contiamo quante di queste erano nello specifico ricerche web
    web_search_count = len([
        msg for msg in messages 
        if isinstance(msg, AIMessage) and msg.tool_calls 
        and any(tc.get("name") == "web_search_tool" for tc in msg.tool_calls)
    ])

    #Se già sono stati invocati 6 tool, verifichiamo se è già stato invocato il source_credibility_tool tra questi
    #Se si, settiamo un flag (has_checked_credibility), altrimenti si attiva l'if succesivo
    if num_tool_calls == 8:
        # Controlliamo in modo nativo se tra i messaggi reali esiste un ToolMessage emesso dal source_credibility_tool
        has_checked_credibility = any(
            isinstance(msg, ToolMessage) and msg.name == "source_credibility_tool" 
            for msg in messages
        )
        
        #Se tra i 6 tool utilizzati, non è stato utilizzato il source_credibility_tool, forziamo tramite prompt ad eseguirlo
        #per evitare di concludere la ricerca senza aver verificato la credibilità delle fonti
        if not has_checked_credibility:
            print("Forzo l'agente a invocare source_credibility_tool prima dello stallo.")
            # Iniettiamo un messaggio di sistema temporaneo per obbligare l'LLM a controllare le fonti ORA
            messages.append(SystemMessage(content=(
                "ATTENZIONE: Hai a disposizione l'ultimo turno utile. Non effettuare nuove ricerche web esterne. "
                "Invocare IMMEDIATAMENTE il 'source_credibility_tool' raccogliendo TUTTI gli URL trovati finora "
                "in un'unica stringa separata da virgole, oppure dichiara 'STATUS: COMPLETE'."
            )))

    # Se arriviamo a 7 chiamate, aggiorniamo il research_status per uscire dal ciclo di ricerca, per evitare di saturare i token 
    if num_tool_calls >= 9:
        print(f"Raggiunto il tetto massimo di {num_tool_calls} chiamate. Chiusura di sicurezza.")
        return {
            "research_status": "COMPLETE"
        }

    #recuperiamo il campo current_target_topic dello stato, aggiornato dal planner e successivamente dall'incremento dell'indice nel graph.py
    topic_info = state.get("current_target_topic", {})
    topic = topic_info.get("topic", "Generico")
    post_type = topic_info.get("type", "Informativo")
    
    #Configurazione dell'LLM a temperatura 0
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0,
    )
    
    # Mappiamo i tool disponibili affinché l'LLM possa vederli e selezionarli dinamicamente
    tools_list = [web_search_tool, rag_retrieval_tool, query_knowledge_graph_tool, source_credibility_tool, web_content_moderator_tool]
    llm_with_tools = llm.bind_tools(tools_list)
    
    

    #Ricostruzione della memoria storica dei messaggi del thread corrente
    # LangGraph accumula in state["messages"] l'intera conversazione ReAct attuale.
    # Se il nodo viene richiamato dopo una revisione umana o un resume non ancora
    # dentro un ciclo tool, ripartiamo con un prompt pulito per evitare sequenze
    # Gemini non valide.
    
    # Controlla se l'ultimo messaggio registrato è un ToolMessage (Garantisce la corretta alternanza della sequenza di chat di Gemini)
    has_active_tool_turn = bool(messages) and isinstance(messages[-1], ToolMessage)

    system_instruction = RESEARCHER_SYSTEM_PROMPT

    # Se il grafo ha estratto un contesto dal KG nei passaggi precedenti, lo iniettiamo per il K-RAG
    kg_context = state.get("context_knowledge_graph", {})
    if kg_context and kg_context.get("exists", False):
        system_instruction += f"\n\n[K-RAG CONTEXT] Informazioni correlate estratte dal Knowledge Graph:\n{kg_context}"

    user_instruction = (
        f"Conduci la ricerca per il seguente post.\n"
        f"Topic: {topic}\n"
        f"Tipologia richiesta: {post_type}\n"
        f"Usa i tool a disposizione per raccogliere dati, espandere le query basandoti sul contesto KG (K-RAG) "
        f"e verificare la credibilità delle fonti. Quando hai terminato e raccolto abbastanza prove "
        f"fondate (grounded), scrivi esplicitamente 'STATUS: COMPLETE' senza invocare altri tool."
    )

    prompt_messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_instruction)
    ]
    # estende la cronologia dei messaggi accumulata nello stato per darla in pasto alla memoria dell'LLM
    if has_active_tool_turn:
        prompt_messages.extend(messages)

    messages = prompt_messages
        
    #Invocazione dell'LLM (Genera il Thought ed eventualmente la tool call)
    response = llm_with_tools.invoke(messages)

    #a partire da response.content (contenuto della risposta dell'llm) vogliamo gestire i vari formati della risposta restituita
    #per convertirla in una stringa
    response_text = response.content
    if isinstance(response_text, list):
        # Se Gemini risponde con una lista strutturata di parti, le unisce inserendo uno spazio di separazione
        response_text = " ".join(str(part) for part in response_text)
    elif response_text is None:
        response_text = ""
    else:
        response_text = str(response_text)
    
    
    # Controlliamo se l'LLM ha autonomamente deciso di terminare la ricerca
    research_status = "IN_PROGRESS" #impostato di default
    #Se è presente la response_text, e contiene al suo interno la stringa "STATUS: COMPLETE" allora poniamo il research_status=COMPLETE
    #per indirizzare il workflow verso il writer
    if response_text and "STATUS: COMPLETE" in response_text.upper():
        print("-> [RESEARCH COMPLETE] L'agente ha terminato autonomamente la raccolta dati.")
        research_status = "COMPLETE"
        
    # Impediamo loop infiniti: se l'agente non chiama tool e non dichiara il completamento, chiudiamo in sicurezza
    if not response.tool_calls and research_status == "IN_PROGRESS":
        print("Nessun tool chiamato. Forzo la chiusura della ricerca per evitare stalli.")
        research_status = "COMPLETE"
        
    # Restituiamo l'aggiornamento a LangGraph. 
    # add_messages si occuperà di fare l'append di questo messaggio nella lista
    return {
        "messages": [response], #con add_message (inserito nello state.py) il messaggio verrà aggiunto, e non sovrascrive i precedenti
        "research_status": research_status
    }