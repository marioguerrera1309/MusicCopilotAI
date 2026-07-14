import sys
import os
import threading #libreria per l'esecuzione parallela in background (Multithreading), necessaria per non bloccare la GUI
import json #per caricare e decodificare le stringhe in formato json
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv #per leggere le variabili d'ambiente dal file .env
from neo4j import GraphDatabase #per connettersi e interrogare il database neo4j
from uuid import uuid4 #per generare id alfanumerici univoci per le sessioni di chat
import tkinter as tk #libreria Tkinter per creare l'interfaccia grafica
from tkinter import ttk, messagebox, scrolledtext #importa i widget moderni, le finestre popup di dialogo e le aree con barre per scorrimento del testo
from langchain_core.messages import ToolMessage, AIMessage #per riconoscere e mappare i messaggi emessi dal workflow di LangGraph

# Allineamento dei path per raggiungere la cartella src e config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from config.config import project_root
from src.graph import compiled_graph #istanza del workflow compiled in graph.py

#Classe principale dell'interfaccia grafica
class BlogCopilotGUI: 
    def __init__(self, root):
        self.root = root
        #Definisce il titolo dell'applicazione, visibile in alto
        self.root.title("CCAI Project 2026 - Autonomous Agentic Blog Copilot (Neo4j Backend)")
        #Dimensioni della finestra (altezza * larghezza in pixel)
        self.root.geometry("1350x850")
        #sfondo grigio chiaro tra le varie aree di testo e pulsanti
        self.root.configure(bg="#f4f6f9")
        
        #Inizializziamo i canali di stato per tracciare il thread corrente, la configurazione e il checkpoint del grafo
        self.current_thread_id = None
        self.config = None
        self.graph_state = None

        self._build_ui() #Chiama il metodo interno _build_ui()
        
    #Metodo costruttore dell'interfaccia grafica    
    def _build_ui(self):
        style = ttk.Style()
        # Seleziona il tema grafico "clam" per garantire l'omogeneità estetica tra Windows, macOS e Linux
        style.theme_use("clam")
        # Configura l'aspetto predefinito dei bottoni della GUI impostando carattere e margini di spaziatura (padding)
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        
        self.root.columnconfigure(0, weight=1) # La colonna di sinistra si allarga per mostrare i log e la bozza
        self.root.columnconfigure(1, weight=0) # La colonna di destra rimane fissa per ospitare la barra laterale
        self.root.rowconfigure(0, weight=0) # Pannello superiore fisso
        self.root.rowconfigure(1, weight=1) # Pannello centrale elastico (Log e Bozza si allungano in verticale)
        self.root.rowconfigure(2, weight=0) # Pannello centrale elastico (Log e Bozza si allungano in verticale)
        self.root.rowconfigure(3, weight=0) # Spaziatore finale
        
        #Definiamo un pannello superiore per iniziare la pianificazione
        top_frame = ttk.Frame(self.root, padding=10)
        # Posiziona il pannello nella prima riga agganciandolo da est a ovest per riempire la larghezza
        top_frame.grid(row=0, column=0, sticky="ew")
        
        #Assegniamo il titolo al pannello di avvio ricerca
        ttk.Label(top_frame, text="Regia Editoriale Agentica Autonoma:", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
        # Testo di stato informativo dinamico che comunica all'utente l'azione corrente del sistema
        self.status_label = ttk.Label(top_frame, text="Pronto ad analizzare i gap di copertura su Neo4j per proporre nuovi contenuti.", font=("Segoe UI", 10, "italic"), foreground="#2c3e50")
        self.status_label.pack(anchor=tk.W, pady=5)
        
        top_btn_container = ttk.Frame(top_frame)
        top_btn_container.pack(fill=tk.X, pady=2)
        
        # Bottone principale per avviare il flusso da zero chiamando il Planner
        self.start_btn = ttk.Button(top_btn_container, text="Avvia Pianificazione e Ricerca Autonoma", command=self.start_agent_flow)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        #Bottone per visualizzare il database a grafo
        self.view_kg_btn = ttk.Button(top_btn_container, text="Visualizza Grafo Real-Time (Neo4j)", command=self.show_graph_popup)
        self.view_kg_btn.pack(side=tk.RIGHT, padx=2)
        
        # Pannello centrale
        mid_frame = ttk.Frame(self.root, padding=10)
        mid_frame.grid(row=1, column=0, sticky="nsew")
        
        # Scompartimento Sinistro: Traccia del ragionamento
        left_frame = ttk.Frame(mid_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        ttk.Label(left_frame, text="Traccia del Ragionamento Autonomo (ReAct - LangSmith Mode):", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        # Campo di testo speciale con barra di scorrimento incorporata e sfondo scuro stile Terminale/IDE
        self.log_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Consolas", 10), bg="#1a202c", fg="#cbd5e0")
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Configurazione tag di formattazione per lo stile LangSmith Run
        self.log_text.tag_configure("HEADER", foreground="#63b3ed", font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("THOUGHT", foreground="#e9d8a6", font=("Consolas", 10, "italic"))
        self.log_text.tag_configure("ACTION", foreground="#f4a261", font=("Consolas", 10, "bold"))
        self.log_text.tag_configure("QUERY", foreground="#38b2ac", font=("Consolas", 10))
        self.log_text.tag_configure("OBSERVATION", foreground="#b794f4", font=("Consolas", 10))
        self.log_text.tag_configure("DATA_OUTPUT", foreground="#cbd5e0", font=("Consolas", 9))
        self.log_text.tag_configure("SUCCESS", foreground="#48bb78", font=("Consolas", 10, "bold"))
        
        #Scompartimento destro per la bozza finale
        right_frame = ttk.Frame(mid_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        ttk.Label(right_frame, text="Bozza dell'Articolo Generato (K-RAG Grounding):", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        self.draft_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Segoe UI", 11), bg="#ffffff", fg="#2d3748")
        self.draft_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Pannello per l'interazione con l'utente
        self.hitl_frame = ttk.LabelFrame(self.root, text=" Validazione Umana della Sequenza ", padding=15)
        self.hitl_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        ttk.Label(self.hitl_frame, text="Note di modifica per la riscrittura (opzionale):").pack(anchor=tk.W)
        # Campo di input testuale ad una riga in cui l'utente può inserire le correzioni (Notes) da passare al Writer
        self.feedback_entry = ttk.Entry(self.hitl_frame, font=("Segoe UI", 11))
        self.feedback_entry.pack(fill=tk.X, pady=5)
        
        btn_container = ttk.Frame(self.hitl_frame)
        btn_container.pack(fill=tk.X, pady=5)
        
        # I bottoni HITL partono tutti in modalità tk.DISABLED (Spenti) e si accenderanno solo quando il grafo si congela
        self.approve_btn = tk.Button(btn_container, text="APPROVA POST (Salva nel DB a Grafo)", bg="#2ecc71", fg="white", font=("Segoe UI", 10, "bold"), command=self.action_approve, state=tk.DISABLED)
        self.approve_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.modify_btn = tk.Button(btn_container, text="RICHIEDI MODIFICA", bg="#f1c40f", fg="black", font=("Segoe UI", 10, "bold"), command=self.action_modify, state=tk.DISABLED)
        self.modify_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.reject_btn = tk.Button(btn_container, text="RESPINGI E RIGENERA", bg="#e74c3c", fg="white", font=("Segoe UI", 10, "bold"), command=self.action_reject, state=tk.DISABLED)
        self.reject_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Pannello laterale destro in cui sono mostrati i post da pubblicare e i post pubblicati
        sidebar_frame = ttk.Frame(self.root, padding=0) 
        sidebar_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=(5, 10), pady=10)
        
        # Sezione "Post pubblicati"
        published_frame = ttk.LabelFrame(sidebar_frame, text=" Post pubblicati nell'archivio storico ", padding=5)
        published_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.published_text = scrolledtext.ScrolledText(
            published_frame, 
            wrap=tk.WORD, 
            font=("Segoe UI", 10), 
            bg="#ffffff", 
            fg="#2d3748",
            width=35  
        )
        self.published_text.pack(fill=tk.BOTH, expand=True)
        self.published_text.insert(tk.END, "- Nessun post pubblicato al momento.")
        self.published_text.config(state=tk.DISABLED) #blocca la scrittura manuale da parte dell'utente

        # Sezione "Post da pubblicare"
        self.to_publish_frame = ttk.LabelFrame(sidebar_frame, text=" Post da pubblicare ", padding=5)
        self.to_publish_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.to_publish_text = scrolledtext.ScrolledText(
            self.to_publish_frame, 
            wrap=tk.WORD, 
            font=("Segoe UI", 10), 
            bg="#ffffff", 
            fg="#2d3748",
            width=35  
        )
        self.to_publish_text.pack(fill=tk.BOTH, expand=True)
        self.to_publish_text.insert(tk.END, "- Nessun post in coda.")
        self.to_publish_text.config(state=tk.DISABLED)
        
        #All'avvio dell'app, viene aperto il file publication_state.json per estrarre i file da pubblicare 
        try:
            with open(os.path.join(os.path.dirname(__file__), "publication_state.json"), "r", encoding="utf-8") as file:
                dati_json_estratti = json.load(file)
            self.carica_coda_da_json(dati_json_estratti)
        except Exception:
            pass
            
        self.carica_post_pubblicati_da_neo4j()

    def carica_coda_da_json(self, dati_json):
        ultimo_controllo = dati_json.get("ultimo_controllo", "Data sconosciuta")
        coda_post = dati_json.get("queue", [])

        # Sottofunzione interna per riallineare in sicurezza la modifica del testo del pannello sul thread principale
        def _update_title():
            nuovo_titolo = f" Post da pubblicare - Ultimo controllo: {ultimo_controllo} "
            self.to_publish_frame.config(text=nuovo_titolo)
        self.root.after(0, _update_title) #chiama la funzione appena definita per aggiornare il titolo

        if not coda_post:
            testo_formattato = "- Nessun post in coda."
        else:
            testo_formattato = ""
            for indice, post in enumerate(coda_post, 1):
                titolo = post.get("title", "Titolo non disponibile")
                contenuto = post.get("content", "Contenuto non disponibile")
                
                testo_formattato += f"[Post {indice}] =========================================\n"
                testo_formattato += f"TITOLO: {titolo}\n"
                testo_formattato += f"--------------------------------------------------\n"
                testo_formattato += f"{contenuto}\n\n\n"

        #Richiamiamo la funzione che aggiungera il testo formattato contentne i post del publication_state.json alla finestra apposita
        self.update_sidebar_content(self.to_publish_text, testo_formattato)

    #Per caricare i post publicati su neo4j, dobbiamo instaurare una connessione e interrogare il database
    def carica_post_pubblicati_da_neo4j(self):
        load_dotenv(project_root() / ".env", override=True)

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not user or not password:
            return

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("MATCH (p:Post) RETURN p.title AS title, p.content AS content LIMIT 20")
                records = list(result)
            driver.close()

            if not records:
                testo_formattato = "- Nessun post pubblicato trovato in Neo4j."
            else:
                testo_formattato = ""
                for indice, record in enumerate(records, 1):
                    titolo = record["title"] if record["title"] else "Titolo non disponibile"
                    contenuto = record["content"] if record["content"] else "Contenuto non disponibile"

                    testo_formattato += f"[Post {indice}] =========================================\n"
                    testo_formattato += f"TITOLO: {titolo}\n"
                    testo_formattato += f"--------------------------------------------------\n"
                    testo_formattato += f"{contenuto}\n\n\n"

            self.update_sidebar_content(self.published_text, testo_formattato)

        except Exception as e:
            errore_msg = f"- Errore di connessione a Neo4j:\n{str(e)}"
            self.update_sidebar_content(self.published_text, errore_msg)
    
    
    # Riapre temporaneamente il widget blindato, cancella il testo vecchio, inserisce il nuovo e lo ricongela
    def update_sidebar_content(self, text_widget, new_content):
        def _update():
            text_widget.config(state=tk.NORMAL)
            text_widget.delete('1.0', tk.END)
            text_widget.insert(tk.END, new_content)
            text_widget.config(state=tk.DISABLED)
        self.root.after(0, _update)

    def log_with_tag(self, message, tag_name=None):
        def _log():
            # Se la funzione riceve uno dei tag configurati (es: 'THOUGHT'), applica l'esatto colore associato
            if tag_name:
                self.log_text.insert(tk.END, message + "\n", tag_name)
            else:
                self.log_text.insert(tk.END, message + "\n")
            # Forza lo scorrimento automatico dell'area di testo (Autoscroll) seguendo l'ultima riga inserita
            self.log_text.see(tk.END)
        self.root.after(0, _log)

    # Alias semplificato per log standard senza colore
    def log(self, message):
        self.log_with_tag(message, None)

    # Metodo per inserire il testo dell'articolo nel pannello di destra della GUI
    def update_draft(self, draft_content):
        def _update():
            self.draft_text.delete('1.0', tk.END)
            self.draft_text.insert(tk.END, draft_content)
        self.root.after(0, _update)

    # Metodo per accendere o spegnere i tre pulsanti di interazione umana
    def set_buttons_state(self, state):
        def _set():
            self.approve_btn.config(state=state)
            self.modify_btn.config(state=state)
            self.reject_btn.config(state=state)
        self.root.after(0, _set)


    def start_agent_flow(self):
        # Genera un ID di sessione univoco per questo specifico thread di LangGraph usando un UUID compatto
        self.current_thread_id = f"autonomous_session_{uuid4().hex[:8]}"
        # Prepara il dizionario di configurazione necessario a MemorySaver per gestire i checkpoint storici
        self.config = {"configurable": {"thread_id": self.current_thread_id}}
        
        # Disabilita il bottone di avvio per impedire all'utente di cliccarlo più volte creando thread sovrapposti
        self.start_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Stato: L'agente sta leggendo Neo4j per scovare gap di copertura...", foreground="#e67e22")
        
        # Svuota i campi di testo per prepararsi a visualizzare i log della nuova sessione
        self.log_text.delete('1.0', tk.END)
        self.update_draft("")
        
        self.log_with_tag("╔════════════════════════════════════════════════════════════════════╗", "HEADER")
        self.log_with_tag("║                 AVVIO REGIA EDITORIALE AUTONOMA                    ║", "HEADER")
        self.log_with_tag("╚════════════════════════════════════════════════════════════════════╝", "HEADER")
        
        #Avvia l'esecuzione del workflow del grafo dentro un Thread parallelo.
        # Questo fa sì che i cicli asincroni girino in background
        threading.Thread(
            target=self._run_graph,
            daemon=True
        ).start()

    def _print_latest_traces(self, node_name, node_update):
        """
        Parser avanzato ad altissima fedeltà (Stile LangSmith completo). 
        Intercetta ed estrae i dati reali contenuti nei ToolMessage nativi emessi dal workflow.
        """
        try:
            if not node_update or not isinstance(node_update, dict):
                return

            #Intercettazione dei dati del planner
            if "planner" in node_name.lower() or "plan" in node_name.lower():
                proposed = node_update.get("proposed_posts", [])
                if proposed and isinstance(proposed, list):
                    self.log_with_tag("\n[PLANNER OUTPUT] -> Nuova strategia editoriale pianificata:", "SUCCESS")
                    for i, post in enumerate(proposed, 1):
                        topic = post.get("topic") or post.get("title") or str(post)
                        reasoning = post.get("justification") or post.get("gap_analizzato") or "Gap rilevato nell'ontologia del grafo."
                        self.log_with_tag(f" Post {i}: {topic}", "SUCCESS")
                        self.log_with_tag(f" ↳ justification: {reasoning}", "DATA_OUTPUT")
                    return

            #Intercettazione dei messaggi della cronologia (Thought,action del researcher e observation dei tool)
            messages = node_update.get("messages", [])
            if messages:
                # Lavoriamo sulla lista dei messaggi emessi in questo specifico update
                msg_list = messages if isinstance(messages, list) else [messages]
                
                for msg in msg_list:
                    # CASO A: Il messaggio arriva da un TOOL (La risposta reale dello strumento)
                    if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
                        t_name = getattr(msg, "name", "Tool")
                        t_content = str(msg.content).strip()
                        
                        self.log_with_tag(f"[Observation ({t_name})]: Risposta ricevuta dallo strumento.", "OBSERVATION")
                        
                        # Pulizia e formattazione dell'output reale del database o del web
                        lines = [l.strip() for l in t_content.split("\n") if l.strip()]
                        if lines:
                            self.log_with_tag("   ┌───[ DATI REALI ACQUISITI DAL TOOL ]───", "OBSERVATION")
                            for l in lines:  
                                self.log_with_tag(f"   │ {l}", "DATA_OUTPUT")
                            self.log_with_tag("   └───────────────────────────────────────", "OBSERVATION")
                        return

                    # CASO B: Chiamata dell'LLM verso un Tool (Action)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for call in msg.tool_calls:
                            t_name = call.get("name", "Sconosciuto")
                            args = call.get("args", {})
                            query_used = args.get("query") or args.get("topic") or args.get("topic_name") or args.get("source_url_or_text") or str(args)
                            
                            text_content = getattr(msg, "content", "")
                            if isinstance(text_content, list):
                                text_content = " ".join([block.get("text", "") for block in text_content if isinstance(block, dict)])
                            
                            text_content = str(text_content).strip()
                            if text_content and not text_content.startswith("["):
                                clean_thought = text_content.replace("THOUGHT:", "").replace("ACTION:", "").strip()
                                self.log_with_tag(f"\n🧠 [Thought]: {clean_thought}", "THOUGHT")
                            
                            self.log_with_tag(f"[Action] -> Invocazione strumento: '{t_name}'", "ACTION")
                            self.log_with_tag(f"[Input Query] -> \"{query_used}\"", "QUERY")
                        return

                    # CASO C: Messaggio standard dell'LLM (Pensiero discorsivo)
                    if hasattr(msg, "content") and msg.content:
                        content_text = str(msg.content).strip()
                        if content_text and not content_text.startswith("[") and not content_text.startswith("{"):
                            if "thought:" in content_text.lower() or "action:" in content_text.lower():
                                for line in content_text.split("\n"):
                                    if line.lower().startswith("thought:"):
                                        self.log_with_tag(f"[Thought]: {line[8:].strip()}", "THOUGHT")
                                    elif line.lower().startswith("action:"):
                                        self.log_with_tag(f"[Action]: {line[7:].strip()}", "ACTION")
                                    elif line.strip():
                                        self.log_with_tag(f"   {line.strip()}", "DATA_OUTPUT")
                            else:
                                if "status: complete" in content_text.lower():
                                    self.log_with_tag(f"✨ [AI Status]: {content_text}", "SUCCESS")
                                else:
                                    self.log_with_tag(f"✉️  [AI Output]: {content_text}", "DATA_OUTPUT")
                            return

            #Accediamo ai campi dello stato 
            for state_key in ["context_knowledge_graph", "recovered_documents_RAG", "research_status"]:
                if state_key in node_update:
                    data = node_update[state_key]
                    if data:
                        self.log_with_tag(f"[Grounding Context ({state_key})]: Aggiornamento dizionario locale.", "OBSERVATION")
                        str_data = str(data)
                        self.log_with_tag(f"   ↳ {str_data}...", "DATA_OUTPUT")

        except Exception as e:
            print(f"DEBUG GUI: Errore nel parsing della traccia live per il nodo {node_name}: {e}")

    def _run_graph(self):
        initial_state = {
            "proposed_posts": [],           
            "current_post_index": 0,
            "current_target_topic": {},
            "context_knowledge_graph": {},
            "recovered_documents_RAG": [], 
            "reasoning_trace": [],
            "research_status": "",
            "generated_draft": "",
            "feedback": "",
            "notes": "",
            "approved_posts": [],
            "publication_schedule": [],
            "published_posts": [],
            "posts_to_publish": [],
            "publication_summary": ""
        }
        
        try:
            # Sfrutta il metodo .stream() in modalità "updates" per catturare l'output di ogni singolo nodo non appena finisce l'azione
            for event in compiled_graph.stream(initial_state, self.config, stream_mode="updates"):
                for node_name, update in event.items():
                    # Disegna un banner grafico di demarcazione nel log per segnalare l'avvio del nodo corrente
                    self.log_with_tag(f"\n┌──────────────────────────────────────────────────────────────────┐", "HEADER")
                    self.log_with_tag(f"  ▶ RUNNING NODE: {node_name.upper()}", "HEADER")
                    self.log_with_tag(f"└──────────────────────────────────────────────────────────────────┘", "HEADER")
                    # Chiama la funzione di tracciamento e colorazione scritta sopra
                    self._print_latest_traces(node_name, update)
            
            # Una volta completato lo stream, se il grafo incontra interrupt_before=["human_review"], il ciclo si blocca.
            # Recuperiamo lo stato del thread congelato usando .get_state()
            self.graph_state = compiled_graph.get_state(self.config)
            vals = self.graph_state.values if self.graph_state else {}
            
            # Estrae la bozza generata dal Writer e il topic corrente per mostrarli sulla GUI
            draft = vals.get("generated_draft", "Nessuna bozza generata.")
            topic_corrente = vals.get("current_target_topic", {}).get("topic", "Sconosciuto")
            
            # Aggiorna i widget grafici riallineandoli in modalità Thread-Safe ed attiva i tre pulsanti HITL (tk.NORMAL)
            self.root.after(0, lambda: self.status_label.config(text=f"Post Corrente: {topic_corrente} (In attesa di revisione)", foreground="#2ecc71"))
            self.update_draft(draft)
            self.set_buttons_state(tk.NORMAL)
            
        except Exception as e:
            self.log_with_tag(f"\n[ERRORE CRITICO DI RUNTIME]: {str(e)}", "ACTION")
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

    #Attivata se l'utente preme il pulsante di approvazione
    def action_approve(self):
        self._disable_hitl_buttons() #disabilitiamo gli altri pulsanti
        self.log_with_tag("\n[HITL]: Bozza approvata dall'utente.", "SUCCESS")
        compiled_graph.update_state(self.config, {"feedback": "approved", "notes": ""}) #aggironiamo i relativi campi dello stato
        threading.Thread(target=self._resume_graph, daemon=True).start() #riprendiamo l'esecuzione dei thread

    #Attivata se l'utente preme il pulsante di modifica
    def action_modify(self):
        notes = self.feedback_entry.get().strip() #recupera il testo inserito nell'apposito campo
        if not notes:
            messagebox.showwarning("Nota vuota", "Fornisci istruzioni per la modifica.")
            return
        self._disable_hitl_buttons()
        self.log_with_tag(f"\n[HITL]: Richiesta riscrittura parziale. Note inviate al prompt: '{notes}'", "QUERY")
        compiled_graph.update_state(self.config, {"feedback": "modified", "notes": notes}) #stavolta modifica pure il campo notes per permettere al writer la modifica
        threading.Thread(target=self._resume_graph, daemon=True).start()

    #Invocata se l'utente preme il pulsante di modifica
    def action_reject(self):
        self._disable_hitl_buttons()
        self.log_with_tag("\n[HITL]: Bozza respinta. Riavvio dell'esplorazione investigativa e dei tool.", "ACTION")
        compiled_graph.update_state(self.config, {"feedback": "rejected", "notes": ""})
        threading.Thread(target=self._resume_graph, daemon=True).start()

    def _resume_graph(self):
        try:
            # Passando None come primo argomento, .stream() capisce che deve ripartire dall'esatto punto in cui era congelato
            for event in compiled_graph.stream(None, self.config, stream_mode="updates"):
                for node_name, update in event.items():
                    self.log_with_tag(f"\n┌──────────────────────────────────────────────────────────────────┐", "HEADER")
                    self.log_with_tag(f"  ▶ RESUMING NODE: {node_name.upper()}", "HEADER")
                    self.log_with_tag(f"└──────────────────────────────────────────────────────────────────┘", "HEADER")
                    self._print_latest_traces(node_name, update)
            
            # Ispezione dello stato dopo il risveglio
            self.graph_state = compiled_graph.get_state(self.config)
            vals = self.graph_state.values if self.graph_state else {}
            
            #Se il grafo si è ricongelato (perché siamo passati all'articolo #2 o #3 della sequenza)
            if self.graph_state and self.graph_state.next:
                draft = vals.get("generated_draft", "")
                topic_corrente = vals.get("current_target_topic", {}).get("topic", "Sconosciuto")
                self.root.after(0, lambda: self.status_label.config(text=f"Post Corrente: {topic_corrente} (In attesa di revisione)", foreground="#2ecc71"))
                self.update_draft(draft)
                self.set_buttons_state(tk.NORMAL) # Riaccende i bottoni per la nuova revisione del post successivo
            else:
                # SCENARIO FINALE: Tutti e 3 i post della pianificazione editoriale sono stati scritti e approvati
                self.log_with_tag("\n╔════════════════════════════════════════════════════════════════════╗", "HEADER")
                self.log_with_tag("║                 CICLO DI COPERTURA CONCLUSO CON SUCCESSO           ║", "HEADER")
                self.log_with_tag("╚════════════════════════════════════════════════════════════════════╝", "HEADER")
                # Riaccende il bottone principale
                self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_label.config(text="Stato: Tutti i post in sequenza sono consolidati su Neo4j.", foreground="#2c3e50"))
                
                try:
                    with open(os.path.join(os.path.dirname(__file__), "publication_state.json"), "r", encoding="utf-8") as file:
                        dati_json_estratti = json.load(file)
                    self.carica_coda_da_json(dati_json_estratti)
                except Exception:
                    pass
                self.carica_post_pubblicati_da_neo4j()
                    
        except Exception as e:
            self.log_with_tag(f"\n[ERRORE NEL RESUME WORKFLOW]: {str(e)}", "ACTION")
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        finally:
            self.root.after(0, lambda: self.feedback_entry.delete(0, tk.END))

    def _disable_hitl_buttons(self):
        self.set_buttons_state(tk.DISABLED)

    def show_graph_popup(self):
        

        uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        # Istanzia un oggetto Grafo Diretto (DiGraph) fornito dalla libreria NetworkX
        G = nx.DiGraph()
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                # Estrae live un campione composto dalle relazioni presenti dentro Neo4j
                result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
                for record in result:
                    n, m = record["n"], record["m"]
                    # Estrae l'etichetta del nodo sorgente e del nodo destinazione (es: 'Post', 'Claim')
                    label_n = list(n.labels)[0] if n.labels else "Unknown"
                    label_m = list(m.labels)[0] if m.labels else "Unknown"
                    # Aggiunge i nodi all'oggetto NetworkX mappando l'element_id interno e prelevando i nomi testuali reali
                    G.add_node(n.element_id, label=label_n, name=n.get("name") or n.get("title") or n.get("text", "Nodo"))
                    G.add_node(m.element_id, label=label_m, name=m.get("name") or m.get("title") or m.get("text", "Nodo"))
                    # Traccia l'arco (relazione) registrando la tipologia della relazione Cypher (es: 'WROTE', 'COVERS')
                    G.add_edge(n.element_id, m.element_id, type=record["r"].type)
            driver.close()
        except Exception as e:
            messagebox.showerror("Errore DB", f"Impossibile leggere da Neo4j: {e}")
            return

        # Controllo di presenza nodi: se il database è vuoto (primo avvio assoluto), blocca il popup informando l'utente
        if not G.nodes:
            messagebox.showinfo("Grafo Vuoto", "Neo4j è attivo ma non contiene ancora nodi.")
            return

        # Inizializza la finestra di disegno di Matplotlib
        plt.figure(figsize=(10, 7))
        plt.title("Vista Real-Time Ontologia Editoriale da Neo4j")
        # Calcola la disposizione spaziale dei nodi applicando l'algoritmo Spring Layout (Modello a Molle per distanziare i nodi)
        pos = nx.spring_layout(G, k=0.4)
        
        colors, labels = [], {}
        # Assegna a ciascun nodo un colore specifico e categorizzato basandosi sulla sua etichetta
        for node, data in G.nodes(data=True):
            # Tronca il nome del nodo a 18 caratteri per evitare sovrapposizioni visive nel disegno dei cerchi
            labels[node] = str(data.get("name", ""))[:18]
            l = data.get("label", "")
            if l == "Post": colors.append("#2ecc71")
            elif l == "Claim": colors.append("#e67e22")
            elif l == "Source": colors.append("#9b59b6")
            elif l == "Agent": colors.append("#1abc9c")
            else: colors.append("#3498db")
        #Disegna il grafico con tutti i nodi e archi aggiornati e con tutte le proprietà di colore, carattere, spaziatura e dimensione aggiornate
        nx.draw(G, pos, labels=labels, node_color=colors, node_size=900, font_size=7, font_weight="bold", edge_color="#bdc3c7", arrows=True)
        #proietta a schermo il grafo risultante, impostando block=False affinché l'utente possa muovere il grafico senza bloccare la GUI Tkinter
        plt.show(block=False)

if __name__ == "__main__":
    # Inizializza l'istanza principale del motore grafico Tkinter
    root = tk.Tk()
    # Costruisce e avvia l'interfaccia accoppiandola alla root
    app = BlogCopilotGUI(root)
    # Lancia l'event loop per mantenere l'applicazione accesa e in ascolto dei click dell'utente
    root.mainloop()