import os
import sys
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

# Allineamento dei path per raggiungere i moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.state import AgentState
from config.prompts import WRITER_SYSTEM_PROMPT
from config.config import get_google_api_key

def _extract_credibility_score(docs_context: str, url: str) -> str:
    """
    Scansiona l'intero contesto dei documenti alla ricerca del rating assegnato
    dal source_credibility_tool per uno specifico URL tramite Regex globale.
    """
    if not docs_context or not url:
        return "Verificata"
        
    # Failsafe immediato: se l'URL fa riferimento alla conoscenza locale o RAG,
    # non cerchiamo nel tool di credibilità e marchiamo come Verificata dall'Admin.
    if "chromadb" in url.lower() or "local" in url.lower() or "sistema-conoscenza" in url.lower():
        return "Verificata (Interna)"
        
    try:
        # Rimuoviamo eventuali parametri di tracciamento o query string (per esempio ?utm_source=...) e puliamo gli spazi
        clean_url = url.split("?")[0].strip().lower()
        # Isoliamo la parte più significativa dell'URL per evitare discrepanze di protocollo (http/https)
        url_pattern = clean_url.replace("https://", "").replace("http://", "").replace("www.", "")
        
        # Trasformiamo tutto il contesto in minuscolo per una ricerca case-insensitive
        context_lower = docs_context.lower()
        
        # Cerchiamo la riga o il blocco che contiene sia l'URL che il punteggio numerico /100.
        # Questa regex isola il blocco di testo della fonte e cattura direttamente le cifre.
        # Cerca la stringa dell'URL e cattura il primo pattern 'XX/100' che incontra subito dopo.
        match_score = re.search(rf"{re.escape(url_pattern)}.*?(\d+)\s*/\s*100", context_lower, re.DOTALL)
        if match_score:
            return f"{match_score.group(1)}/100"
                            
    except Exception as e:
        print(f"[WARNING WRITER PARSING]: Impossibile estrarre lo score per {url}: {e}")
        
    return "Verificata"


def write_post_node(state: AgentState) -> dict:
    """
    Nodo del Writer: Genera la bozza del post blog applicando il pattern K-RAG,
    garantendo la consistenza con i claim passati e inserendo citazioni esplicite.
    """
    print("\n Redazione della bozza dell'articolo (K-RAG) ---")
    
    #preleva il current_target_topic popolato dal Planner e a poco a poco aggiornato nel graph.py
    topic_info = state.get("current_target_topic", {})
    #preleva il context_knowledge_graph restituito dal kg_tool, a sua volta restituitogli dalla funzione get_context_for_topic del kg_manager.py
    kg_context = state.get("context_knowledge_graph", {})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_google_api_key(),
        temperature=0.7, # Mantenuta per dare fluidità e stile alla scrittura editoriale
    )

    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", (
            "Genera ora l'articolo completo seguendo i fatti accertati.\n\n"
            "### DATI RECUPERATI PER LA STESURA ###\n"
            "Topic: {topic}\n"
            "Tipo di Post: {post_type}\n"
            "Contesto Storico (KG): {kg_context}\n"
            "Fonti e Testi Estratti (RAG/Web/Credibilità):\n{docs}\n\n"
            "Note di Modifica dell'Utente:\n{modifications}"
        ))
    ])
    
    
    if isinstance(kg_context, dict) and kg_context.get("exists") == True:
        kg_lines = [f"Ecosistema d'archivio per il tema: {kg_context.get('matched_topic', '')}"]
        if kg_context.get("previous_claims"):
            kg_lines.append(f"Claim passati verificati: {', '.join(kg_context['previous_claims'])}")
        if kg_context.get("previous_full_posts"):
            kg_lines.append("\nArticoli completi approvati riutilizzabili per mantenere la coerenza:")
            for p in kg_context["previous_full_posts"]:
                kg_lines.append(f"--- (Titolo: {p['title']}) ---\n{p['text']}")
        kg_grounding_text = "\n".join(kg_lines)
    else:
        kg_grounding_text = "Questo argomento è un gap di copertura: non esistono articoli pregressi correlati nel Knowledge Graph."

    #Estraiamo i messaggi dalla cronologia del Researcher
    #le seguenti sono liste di dizionari
    web_outputs = []
    credibility_outputs = []
    rag_documents = []
    
    messages_history = state.get("messages") or []
    for msg in messages_history:
        if isinstance(msg, ToolMessage):
            tool_name = msg.name
            tool_content = msg.content
            
            if tool_name == "web_search_tool":
                web_outputs.append({
                    "content": tool_content,
                    "source": "Ricerca Web Esterna (Tavily)"
                })
            elif tool_name == "source_credibility_tool":
                credibility_outputs.append({
                    "content": tool_content,
                    "source": "Report di Credibilità Fonte"
                })
            elif tool_name == "rag_retrieval_tool":
                rag_documents.append({
                    "content": tool_content,
                    "source": "Archivio Interno RAG Vettoriale (ChromaDB)"
                })

    all_documents = rag_documents + web_outputs + credibility_outputs
    print(f"Dati estratti da ReAct messages -> RAG: {len(rag_documents)} | Web: {len(web_outputs)} | Credibilità: {len(credibility_outputs)}")

    # Serializzazione dei blocchi informativi per l'LLM
    #Trasforma la lista di documenti in un'unica stringa leggibile dall'LLM
    docs_serialized = ""
    if all_documents:
        serialized_list = []
        # Cicla sui documenti assegnando una numerazione umana progressiva partendo da 1
        for i, doc in enumerate(all_documents, 1):
            source_name = doc.get('source', 'Sorgente Sconosciuta')
            content_text = doc.get('content', 'Nessun testo recuperato.')
            serialized_list.append(f"-[Fonte #{i} (Origine: {source_name})]:\n{content_text}")
        # Unisce i blocchi inserendo una doppia andata a capo
        docs_serialized = "\n\n".join(serialized_list)
    
    # Recupero delle note dall'Human-in-the-loop
    # Estrae le correzioni scritte dall'utente sulla GUI o su langsmith(salvate nella chiave "notes" dello stato)
    user_modifications = state.get("notes", "").strip() if state.get("notes") else ""
    if not user_modifications:
        user_modifications = "L'utente ha approvato la pianificazione così com'è. Procedi alla stesura dell'articolo senza modifiche strutturali o correzioni aggiuntive."
    
    # Compila il prompt finale inserendo tutte le variabili estratte
    formatted_msg = writer_prompt.format_messages(
        topic=topic_info.get("topic", ""),
        post_type=topic_info.get("type", ""),
        kg_context=kg_grounding_text,
        docs=docs_serialized if docs_serialized else "Nessun documento esterno o locale recuperato durante la fase di esplorazione ReAct.",
        modifications=user_modifications
    )
    
    try:
        response = llm.invoke(formatted_msg)

        print("-> [SUCCESS] Bozza dell'articolo redatta con successo.")
        
        # Estrae la stringa dell'articolo generato dal writer
        final_article = response.content if response.content else ""
        
        
        try:
            # Applica una regex globale per individuare tutti i link in formato Markdown presenti nel testo generato: [Testo](URL)
            markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', final_article)
            # Cicla su ciascuna tupla (testo del link, URL fisico) intercettata nell'articolo
            for text, url in markdown_links:
                # Definisce la stringa standard di controllo fittizia
                stringa_generica = f"({url}) [Credibilità: Verificata]"
                # Verifica Se l'LLM ha inserito la stringa generica o se ha dimenticato del tutto di agganciare il voto tra parentesi
                if stringa_generica in final_article or f"({url}) [" not in final_article:
                    # Chiama la funzione utility scritta in cima per andare a recuperare il punteggio numerico reale /100 nei report
                    real_score = _extract_credibility_score(docs_serialized, url)
                    #Verifica se il punteggio reale estratto è numerico (diverso dal fallback "Verificata")
                    if real_score != "Verificata":
                        # Rimpiazza la stringa generica dell'LLM con la stringa corretta contenente il voto matematico reale
                        final_article = final_article.replace(stringa_generica, f"({url}) [Credibilità: {real_score}]")
                        # Se il link era completamente privo di etichette di credibilità, le appende subito dopo la parentesi chiusa dell'URL
                        if f"({url}) [" not in final_article:
                            final_article = final_article.replace(f"({url})", f"({url}) [Credibilità: {real_score}]")
        except Exception as filter_err:
            print(f"[WARNING] Errore nel filtro post-processing dei punteggi: {filter_err}")

        return {
            "generated_draft": final_article
        }

    except Exception as e:
        print(f" [ERROR] Errore nell'invocazione di google nel Writer: {e}")
        return {
            "generated_draft": f"## Errore di Generazione\nImpossibile generare la bozza a causa di un'anomalia delle API: {e}"
        }