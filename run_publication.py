import os
import sys
import json
import datetime

# Allineamento dei path per raggiungere config e database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database.kg_manager import EditorialKGManager

def simulate_blog_publication_cron():
    """
    Script autonomo eseguito in background (CRON):
    Funge da 'Guardiano del Tempo'. Controlla il JSON e, se è passato almeno 1 giorno,
    chiama add_approved_post per pubblicare il post più vecchio su Neo4j,
    rimuovendolo stabilmente dal file e aggiornando il timestamp.
    """
    print("\n--- [BACKGROUND JOB: CONTROLLO TEMPO E PUBBLICAZIONE] ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Risaliamo di due livelli (da src/database/ a root del progetto) per trovare il JSON
    json_path = os.path.join(current_dir, "publication_state.json")
    oggi = datetime.date.today()
    
    # Apriamo il file publication_state.json e carichiamo il json in buffer_data 
    if not os.path.exists(json_path):
        print("[CRON] Nessun file JSON trovato. Coda vuota, nulla da pubblicare.")
        return
        
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            buffer_data = json.load(f)
    except Exception as e:
        print(f"[CRON ERRORE] Impossibile leggere il file JSON: {e}")
        return
        
    if "queue" not in buffer_data or not buffer_data["queue"]:
        print("[CRON] Nessun post in attesa nel JSON.")
        return

    #Calcola la differenza tra la data attuale e la data presente in ultimo controllo. Se nel json non è presente il campo ultimo controllo, se ne inizializza una di dieci giorni fa
    data_passata = (oggi - datetime.timedelta(days=10)).isoformat()
    ultimo_controllo_str = buffer_data.get("ultimo_controllo", data_passata)
    ultimo_controllo = datetime.date.fromisoformat(ultimo_controllo_str)
    
    giorni_passati = (oggi - ultimo_controllo).days
    print(f"-> Ultimo controllo: {ultimo_controllo} | Oggi: {oggi} | Giorni passati: {giorni_passati}/1")

    published_posts = []
    successo_pubblicazione = False

    #Controlla se è passato almeno un giorno
    if giorni_passati >= 1:
        print("-> [SBLOCCO] Finestra temporale aperta! Estraggo il post più antico.")
        
        # Ordiniamo la lista interna direttamente sul dizionario madre per evitare disallineamenti, dal post più nuovo al post più vecchio
        buffer_data["queue"].sort(key=lambda x: x["saved_at"])
        
        # Estraiamo il post più vecchio rimuovendolo nativamente dalla struttura
        post_da_pubblicare = buffer_data["queue"].pop(0)
        
        # Istanziamo la classe del kg_manager.py per utilizzare la funzione app_approved_post e caricare il post su Neo4J
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
            print(f"-> [NEO4J SUCCESS] Post '{post_da_pubblicare['title']}' iniettato correttamente!")
            
            successo_pubblicazione = True
            published_posts.append(post_da_pubblicare)
            
        except Exception as e:
            print(f"-> [ERRORE CRITICO NEO4J] add_approved_post fallito: {e}")
            # Failsafe: se il database fallisce, reinseriamo il post in cima alla coda nativa
            buffer_data["queue"].insert(0, post_da_pubblicare)
        finally:
            kg_manager.close()
            
        #Aggiorniamo l'ultimo controllo nel json con la data attuale
        if successo_pubblicazione:
            buffer_data["ultimo_controllo"] = oggi.isoformat()
            print(f"-> [TIMER UPDATE] 'ultimo_controllo' aggiornato a oggi: {oggi.isoformat()}")
            
    else:
        print(f"-> [BLOCCO TEMPORALE] Non è ancora passato 1 giorno. Coda congelata.")

    #Sostituiamo il contenuto del publication_state.json con il json aggiornato in buffer_data
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(buffer_data, f, indent=2, ensure_ascii=False)
    
    rimanenti_in_coda = len(buffer_data["queue"])
    print(f"--- [FINE CRON JOB] Rilasciati {len(published_posts)} post su Neo4j. In coda rimangono {rimanenti_in_coda} post.\n")
    print("--- [FINE CRON JOB] ---")
    sys.exit(0)
if __name__ == "__main__":
    simulate_blog_publication_cron()