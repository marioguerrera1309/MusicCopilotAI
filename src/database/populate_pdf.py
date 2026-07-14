import os
import sys
#importiamo il modulo PyPDFLoader per estrarre il testo dai file.pdf (tramite una funzione .load() del modulo)
from langchain_community.document_loaders import PyPDFLoader
#importiamo lo splitter di langhChain che servirà per la suddivisione del test in chunck
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time #  Assicurati che time sia importato!

#Torna indietro di due cartelle, in modo da effettuare la ricerca dei file e la connessione al database partendo dalla root principale
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#Importiamo la classe VectorRAGManager del file vector_manager.py per instaurare una connessione con ChromaDB
from src.database.vector_manager import VectorRAGManager

def ingest_all_pdfs_from_rag_folder():
    """
    Scansiona la cartella 'documenti_RAG', verifica quali file sono già presenti
    nel Vector DB e indicizza ESCLUSIVAMENTE i PDF nuovi, evitando duplicati.
    """
    #identifica il percorso dove risiede questo script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #aggiunge al percorso precedente la root documenti_RAG
    folder_path = os.path.join(current_dir, "documenti_RAG")

    if not os.path.exists(folder_path):
        print(f" [ERRORE] La cartella dei documenti non esiste al percorso: {folder_path}")
        return
    #raccogliamo in una lista tutti i pdf presenti nella cartella documenti_RAG, se terminano con l'estensione .pdf
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f" Nessun file PDF trovato dentro la cartella: {folder_path}")
        return

    print(f" Rilevati {len(pdf_files)} file PDF fisici nella cartella.")
    
    rag_manager = VectorRAGManager()
    
    #creiamo un set per memorizzare i nomi dei file.pdf caricati nel chromaDB. Utilizziamo il set per evitare duplicati 
    already_indexed_files = set()
    try:
        #Recupera l'elenco di tutti i metadati archiviati nel chromaDB
        existing_data = rag_manager.vector_store.get(include=["metadatas"])
        #se il database non è vuoto, e continene la chiave "metadatas", allora cicla su ciascun dizionario di metadati
        if existing_data and "metadatas" in existing_data:
            for meta in existing_data["metadatas"]:
                #Se il metadato è valido e contiene il campo "source" (il nome originale del file), allora carica il nome del file nel set definito in precedenza
                if meta and "source" in meta:
                    already_indexed_files.add(meta["source"])
        print(f" Il Vector DB contiene già informazioni provenienti da {len(already_indexed_files)} file.")
    except Exception as e:
        print(f" [WARNING] Impossibile verificare i file preesistenti nel database: {e}")

    #Creiamo una lista contenente tutti i nomi dei file della cartella documenti_RAG che non sono presenti in already_indexed_files
    files_to_process = [f for f in pdf_files if f not in already_indexed_files]

    if not files_to_process:
        print(" Tutti i PDF nella cartella sono già stati indicizzati. Nessuna operazione necessaria!")
        return

    print(f" {len(files_to_process)} file sono NUOVI e verranno elaborati adesso.")

    #Configuriamo lo splitter importato per la suddivisione del testo estratto in chunck
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, #ogni chunck ha dimensione di 600 caratteri
        chunk_overlap=100, #ci sono sovrapposizioni di 100 caratteri tra un chunck e l'altro, per mantenere il contesto semantico
        length_function=len #si utilizza la funzione lenght per misurare la lunghezza del testo
    )

    for pdf_name in files_to_process:
        full_pdf_path = os.path.join(folder_path, pdf_name)
        print(f"\nINIZIO ELABORAZIONE NUOVO FILE: {pdf_name} ===")

        try:
            loader = PyPDFLoader(full_pdf_path)
            #viene estratto tutto il testo del file.pdf
            raw_documents = loader.load()
            
            chunks = text_splitter.split_documents(raw_documents)
            print(f"Testo scompattato in {len(chunks)} frammenti.")

            print(f"Avvio caricamento incrementale controllato per '{pdf_name}'...")
            
            #Invece di inviare tutto in una volta, con il rischio che vengano superate il numero di richieste al minuto al modulo di embedding,
            #  inviamo i frammenti uno ad uno introducendo la pausa di sicurezza.
            for index, chunk in enumerate(chunks):
                #Per ogni chunck, creiamo un dizionario metadata con il nome del file e il numero della pagina (viene incrementato di 1 perchè il computer legge da 0)
                #Il numero di pagina è salvato in un campo metadata quando viene effettuato lo split (suddivisione in chunk) dalla classe RecursiveCharacterTextSplitter
                metadata = {
                    "source": pdf_name,  
                    "page": chunk.metadata.get("page", 0) + 1
                }
                
                # Chiamiamo add_documents passando un singolo elemento racchiuso in una lista []
                #texts indica il contenuto vero e proprio e metadatas il dizionario definito in precedenza
                rag_manager.add_documents(texts=[chunk.page_content], metadatas=[metadata])
                
                # Stampiamo una stringa ogni 20 frammenti caricati
                if (index + 1) % 20 == 0:
                    print(f"Progresso: {index + 1}/{len(chunks)} frammenti memorizzati...")
                
                # Pausa di 0.75 secondi: garantisce al massimo 80 richieste al minuto (il limite è 100).
                time.sleep(0.75)

            print(f" '{pdf_name}' memorizzato integralmente con successo!")

        except Exception as e:
            print(f"Impossibile indicizzare il file {pdf_name}: {e}")

    print("\nIl Vector DB è stato aggiornato con i nuovi dati!")

if __name__ == "__main__":
    ingest_all_pdfs_from_rag_folder()