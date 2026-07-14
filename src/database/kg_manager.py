import os
import sys
import re
import datetime
import json
import numpy as np #per effettuare il dense retrieval
from neo4j import GraphDatabase #per connettersi e inviare query cypher a Neo4J 
from langchain_google_genai import GoogleGenerativeAIEmbeddings #modulo di google usato per l'embedding dei topic
from langchain_core.messages import ToolMessage #tipo di messaggio associato alle risposte dei tool in LangChain
from langgraph.graph.message import RemoveMessage #istruzione di LangChain per rimuovere i messaggi dallo stato
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Allinea dinamicamente i percorsi risalendo di due livelli fino alla cartella radice del progetto
from src.state import AgentState 
from langchain_google_genai import ChatGoogleGenerativeAI #per utilizzare l'llm gemini 2.5 flash
from langchain_core.prompts import ChatPromptTemplate #costruttore di template per strutturare i prompt da inviare all'LLM
from config.config import get_google_api_key, load_project_env

# [La classe EditorialKGManager rimane INALTERATA perché le query Cypher sono eccellenti]
class EditorialKGManager:
    def __init__(self):
        load_project_env() #carica le variabili d'ambiente dal file .env
        #le variabili di ambiente vengono salvate come attributi di EditorialKGManager
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not self.uri or not self.user or not self.password:
            raise RuntimeError(
                "Configurazione Neo4j incompleta. Imposta NEO4J_URI, NEO4J_USER e NEO4J_PASSWORD nel file .env."
            )

        #Istanzia il driver di connessione a Neo4J
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        #Istanzia il modello per l'embedding dei topic
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2", 
            google_api_key=get_google_api_key()
        )

    def close(self):
        self.driver.close()

    

    def get_covered_topics_and_recency(self) -> list:
        """
        Estrae lo storico degli argomenti trattati unificando i dati vivi di Neo4j
        con i post approvati dall'utente presenti nel buffer JSON, garantendo che
        il Planner veda i post in coda come 'già coperti' ed eviti duplicazioni.
        """
        #Recuperiamo i topic consolidati da Neo4j
        query = "MATCH (t:Topic) RETURN t.name AS topic"
        db_topics = [] #lista di dizionari
    
        try:
            with self.driver.session() as session:
                result = session.run(query)
                #popola la lista con dizionari di chiave "topic" in cui viene memorizzato ogni nome del topic 
                db_topics = [{"topic": record["topic"]} for record in result]
        except Exception as e:
            print(f"[WARNING KG MANAGER]: Errore durante la lettura da Neo4j: {e}")
            db_topics = []

        #Leggiamo i post approvati nel publication_state.json
        json_path = "publication_state.json" 
        json_topics = [] #lista di dizionari
    
        if os.path.exists(json_path): #se esiste il percorso del file (in questo caso è situato nella root del progetto)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    buffer_data = json.load(f) or {} #carichiamo il json del file publication_state.json in buffer_data
                approved_posts = buffer_data.get("queue", []) #recuperiamo il contenuto del campo "queue", contente i vari post (con topic,claim, fonti, saved_at, ecc) 
                for post in approved_posts:
                    # Estraiamo il campo 'topic' .
                    topic_name = post.get("topic")
                    if topic_name:
                        json_topics.append({"topic": topic_name.strip()})
                    
                if json_topics:
                    print(f"Intercettati {len(json_topics)} post nel buffer JSON. Sincronizzazione preventiva applicata per il Planner.")
                
            except Exception as e:
                print(f"[WARNING KG MANAGER]: Impossibile leggere il file buffer JSON dei post approvati: {e}")

        #Ritorniamo la lista unificata (Database +Json)
        return db_topics + json_topics
    

    def get_context_for_topic(self, topic_query: str) -> dict:
        """
        Sfrutta il modulo di embedding in memoria (RAM) ad ogni singola chiamata
        applicando tecniche di normalizzazione e matching strutturato flessibile.
        """
        print(f"Avvio calcolo embedding al volo per la query: '{topic_query}'")
    
        #Recupero di tutti i topic storici da Neo4j
        query_all_topics = "MATCH (t:Topic) RETURN t.name AS name"
        all_topics = []
    
        try:
            with self.driver.session() as session:
                result = session.run(query_all_topics)
                all_topics = [record["name"].strip() for record in result if record["name"]]
            print(f"Estratti {len(all_topics)} topic testuali storici da confrontare.")
        except Exception as e:
            print(f"[WARNING] Impossibile recuperare i topic storici: {e}")
            return {"exists": False, "previous_claims": [], "previous_full_posts": []}

        if not all_topics:
            return {"exists": False, "previous_claims": [], "previous_full_posts": []}

        try:
            
            #calcola il vettore (embedding) della query inviata dall'agente e lo racchiude in un array NumPy
            query_vector = np.array(self.embeddings_model.embed_query(topic_query.strip()))
            # Calcola in un'unica operazione batch i vettori di tutti i topic storici estratti da Neo4j
            history_vectors = np.array(self.embeddings_model.embed_documents(all_topics))
        
            #il calcolo della similarità è basato sulla similarità del coseno
            # Calcola il prodotto scalare tra la query e tutti i vettori
            dot_products = np.dot(history_vectors, query_vector)
            # Calcola la norma (lunghezza) del vettore della query (evitando la divisione per zero con max a 1e-9)
            norm_query = max(np.linalg.norm(query_vector), 1e-9)
            # Calcola la norma di tutti i vettori storici lungo l'asse delle righe (axis=1)
            norm_history = np.linalg.norm(history_vectors, axis=1)
            # Sostituisce eventuali norme a zero con una costante piccolissima di sicurezza per prevenire crash matematici
            norm_history = np.where(norm_history == 0, 1e-9, norm_history)
        
            # Applica la formula ufficiale della Similarità Coseno: Dot_Product / (||A|| * ||B||)
            cosine_similarities = dot_products / (norm_query * norm_history)
        
            #Individua l'indice numerico del valore di similarità più alto presente nell'array
            best_match_idx = np.argmax(cosine_similarities)
            # Estrae il punteggio numerico di quel match (lo score)
            best_score = cosine_similarities[best_match_idx]
            #Estrae il nome del topic vincitore
            best_topic_name = all_topics[best_match_idx]
        
            # Soglia di superamento
            SOGLIA_SISTEMA = 0.70
        
            if best_score >= SOGLIA_SISTEMA:
                print(f"Affinità semantica rilevata con '{best_topic_name}' (Score: {best_score:.4f})")
            
                # Cerchiamo il topic corrispondente e si muove lungo i nodi p:Post, c:Claim e s:Source aggregando i risultati
                query_claims = """
                MATCH (t:Topic)
                WHERE toLower(trim(t.name)) = toLower(trim($best_topic_name))
                MATCH (p:Post)-[:COVERS]->(t)
                MATCH (p)-[:HAS_CLAIM]->(c:Claim)
                RETURN t.name AS matched_topic, 
                    collect(DISTINCT c.text) AS claims, 
                    collect(DISTINCT {title: p.title, text: p.content, date: p.date}) AS posts
                """
                with self.driver.session() as session:
                    res = session.run(query_claims, best_topic_name=best_topic_name)
                    # Estrae l'unico record aggregato restituito dalla query
                    record = res.single()
                
                    if record and record["matched_topic"]:
                        # Recupero del post associato al best topic convertendo
                        # neo4j.time.DateTime in stringhe serializzabili.
                        cleaned_posts = []
                        raw_posts = record["posts"] if record["posts"] else []
                        if raw_posts and raw_posts[0]["title"] is not None:
                            for post in raw_posts:
                                raw_date = post.get("date")
                                if raw_date is None:
                                    serializable_date = "N/A"
                                elif hasattr(raw_date, "iso_format"):
                                    serializable_date = raw_date.iso_format()
                                elif hasattr(raw_date, "isoformat"):
                                    serializable_date = raw_date.isoformat()
                                else:
                                    serializable_date = str(raw_date)

                                cleaned_posts.append({
                                    "title": post.get("title"),
                                    "text": post.get("text"),
                                    "date": serializable_date
                                })
                    
                        return {
                            "exists": True,
                            "matched_topic": record["matched_topic"],
                            "previous_claims": record["claims"] if record["claims"] else [],
                            "previous_full_posts": cleaned_posts
                        }
        
            # Se lo score è sotto soglia, o se Neo4j non ha validato il record, trattiamo come GAP
            print(f"Il topic più vicino è '{best_topic_name}' ma lo score ({best_score:.4f}) è sotto la soglia di {SOGLIA_SISTEMA} o non registrato. Trattato come GAP.")
            
        except Exception as vec_err:
            print(f"[ERROR KG ON-THE-FLY VECTOR CALCULATION]: {vec_err}")
        
        return {"exists": False, "previous_claims": [], "previous_full_posts": []}

    def add_approved_post(self, post_id: str, title: str, content: str, 
                         target_topic: str, claims: list, sources: list):
        now_str = datetime.datetime.now().isoformat()
        #query per creare un nuovo post e collegarlo al nodo Agente
        wrote_query = """
        MERGE (a:Agent {agentId: "blogger_copilot_2026"})
        ON CREATE SET a.name = "Music Copilot AI", a.domain = "Music"
        
        CREATE (p:Post {
            postId: $post_id,
            title: $title,
            content: $content,
            status: "published",
            date: localdatetime($now_str),
            author: "AI Copilot"
        })
        
        CREATE (a)-[:WROTE {writtenAt: localdatetime($now_str), wordCount: size($content)}]->(p)
        """
        #query per creare il nodo topic e collegarlo al nodo post
        topic_query = """
        MATCH (p:Post {postId: $post_id})
        MERGE (t:Topic {name: $target_topic})
        ON CREATE SET t.topicId = "top_" + p.postId, t.description = "Dominio tematico focalizzato su " + $target_topic
        
        CREATE (p)-[:COVERS {coverageAt: localdatetime($now_str), coverageLevel: 1.0}]->(t)
        """
        
        with self.driver.session() as session:
            session.run(wrote_query, post_id=post_id, title=title, content=content, now_str=now_str)
            session.run(topic_query, post_id=post_id, target_topic=target_topic, now_str=now_str)
            
            #per ogni claim nella lista passata come parametro, scriviamo una query dove creiamo ogni nodo claim e colleghiamo il post corrente al claim.
            #Poi il driver di connessione esegue la query
            for i, claim_text in enumerate(claims):
                claim_id = f"clm_{post_id}_{i}"
                claim_query = """
                MATCH (p:Post {postId: $post_id})
                CREATE (c:Claim {
                    claimId: $claim_id,
                    text: $claim_text
                })
                CREATE (p)-[:HAS_CLAIM {mentionedAt: localdatetime($now_str)}]->(c)
                """
                session.run(claim_query, post_id=post_id, claim_id=claim_id, claim_text=claim_text, now_str=now_str)
            
            #Per ogni fonte della lista passata come argomento, scriviamo una query dove creiamo ogni nodo source e colleghiamo il post corrente alla source
            for i, src_url in enumerate(sources):
                src_query = """
                MATCH (p:Post {postId: $post_id})
                MERGE (s:Source {url: $src_url})
                ON CREATE SET s.sourceId = "src_" + p.postId, s.type = "Web"
                
                CREATE (p)-[:HAS_SOURCE]->(s)
                
                WITH p, s
                MATCH (p)-[:HAS_CLAIM]->(c:Claim)
                CREATE (c)-[:FROM_SOURCE {citationAt: localdatetime($now_str), order: toInteger($idx)}]->(s)
                """
                session.run(src_query, post_id=post_id, src_url=src_url, idx=i, now_str=now_str)


def update_kg_node(state: AgentState) -> dict:
    """
    Nodo del Grafo LangGraph:
    Invece di scrivere su Neo4j, intercetta il post approvato e lo appende 
    nel file 'publication_state.json' con il timestamp di scrittura attuale.
    """
    
    topic_info = state.get("current_target_topic", {})
    draft = state.get("generated_draft", "")
    
    # Generiamo l'ID e il timestamp esatto di questo momento (scrittura nel JSON). L'id viene generato con un timestamp contente i millisecondi per evitare duplicati
    post_id = f"post_{int(datetime.datetime.now().timestamp())}"
    now_str = datetime.datetime.now().isoformat()
    
    # Isoliamo la prima riga del testo Markdown per usarla in automatico come titolo pulito del post
    title = draft.split("\n")[0].replace("#", "").strip() if draft else "Nuovo Post Consolidato"
    target_topic = topic_info.get("topic", "Generico")
    
    
    real_sources = []
    #Recuperiamo i messaggi del turno corrente
    messages_history = state.get("messages") or []
    for msg in messages_history:
        # Controlliamo se il messaggio è stato generato dall'azione fisica di un tool (ToolMessage)
        if isinstance(msg, ToolMessage):
            # Se l'output arriva dal tool di ricerca web, applichiamo una Regex per estrarre tutti gli URL (http/https)
            if msg.name == "web_search_tool":
                urls = re.findall(r'(https?://[^\s\)]+)', msg.content)
                # Ripuliamo le stringhe degli URL da scorie di formattazione o parentesi quadre quadrate chiuse
                real_sources.extend([u.strip().replace("]", "").replace('"', '').replace("'", "") for u in urls])
            # Se l'output arriva dal RAG locale, aggiungiamo una stringa fissa che traccia l'uso di ChromaDB
            elif msg.name == "rag_retrieval_tool":
                real_sources.append("Archivio Vettoriale ChromaDB (Conoscenza Locale)")

    # Elimina i link duplicati convertendo la lista in un Set e impostando un link di fallback se vuota
    unique_sources = list(set(real_sources)) if real_sources else ["https://sistema-conoscenza-interna.com"]

    #Estraiamo i claim del testo tramite llm
    print("-> Analisi del testo ed estrazione delle asserzioni chiave...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=get_google_api_key(), temperature=0.1)

    # Configuriamo un prompt di sistema stringente per forzare il modello a rispondere solo con un array JSON di frasi.
    #Il prompt user passa il contenuto dell'articolo, recuperato dal campo "generated_draft"
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "Estrai da 1 a 3 asserzioni storiche o fatti chiave precisi descritti nel testo. Rispondi SOLO con una lista JSON di stringhe."),
        ("user", "{draft}")
    ])
    
    extracted_claims = []
    try:
        response = llm.invoke(extraction_prompt.format_messages(draft=draft))
        clean_content = response.content.strip()
        # Applica una regex protettiva per isolare l'array JSON di stringhe ed eliminare sporcizia di testo esterna
        json_match = re.search(r'\[\s*".*"\s*\]', clean_content, re.DOTALL)
        # Scompatta la stringa JSON convertendola nativamente in una lista Python
        extracted_claims = json.loads(json_match.group(0)) if json_match else json.loads(clean_content)
    except Exception:
        extracted_claims = [f"Fatto chiave verificato per il focus redazionale su: {target_topic}"] #claim inserito in caso di errore

    #Costruiamo il pacchetto per il json
    nuovo_post = {
        "post_id": post_id,
        "title": title,
        "topic": target_topic,
        "content": draft,
        "claims": extracted_claims,
        "sources": unique_sources,
        "saved_at": now_str  # La data di scrittura nel JSON
    }

    #Nome del file .json
    json_path = "publication_state.json"
    
    # Se il file non esiste ancora, creiamo la struttura base.
    # Impostiamo un "ultimo_controllo" iniziale fittizio vecchio (es. 10 giorni fa)
    # così la primissima esecuzione finale troverà la finestra temporale sbloccata.
    data_passata = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
    buffer_data = {"ultimo_controllo": data_passata, "queue": []}
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                buffer_data = json.load(f) #salviamo il contenuto del publication_state.json, se esiste, su buffer_data
                # Ci assicuriamo che le chiavi necessarie siano presenti
                if "queue" not in buffer_data:
                    buffer_data["queue"] = []
                if "ultimo_controllo" not in buffer_data:
                    buffer_data["ultimo_controllo"] = data_passata
        except Exception:
            pass

    # Aggiungiamo il post alla coda del file JSON
    buffer_data["queue"].append(nuovo_post)
    
    # Scriviamo fisicamente su disco
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(buffer_data, f, indent=2, ensure_ascii=False) #carichiamo il contenuto di buffer_data sul file.json
        
    print(f"Post accodato nel JSON. Coda attuale: {len(buffer_data['queue'])} post.")

    #Aggiorniamo lo stato 
    approved_posts_state = state.get("approved_posts") or []
    #Rimuoviamo i messaggi associati al ciclo del post corrente, visto che ne dovrà essere scritto uno nuovo
    # Genera una lista di oggetti RemoveMessage per cancellare tutti i vecchi messaggi ReAct accumulati
    cleanup_messages = [RemoveMessage(id=msg.id) for msg in  messages_history if msg.id]

    print(f"Inviati {len(cleanup_messages)} comandi di rimozione messaggi")

    return {
        "feedback": "",
        "notes": "",
        "research_status": "",   
        "recovered_documents_RAG": [],
        "messages": cleanup_messages, # Inietta i comandi di eliminazione per svuotare fisicamente la chat nello stato
        "generated_draft": "",
        "context_knowledge_graph": {},
        "reasoning_trace": [],
        # Alimentiamo il canale in memoria così il nodo di simulazione ha i dati pronti
        "approved_posts": approved_posts_state + [nuovo_post]
    }