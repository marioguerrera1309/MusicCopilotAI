import os
from pathlib import Path
#importiamo la funzione per caricare le variabili scritte dentro il file .env
from dotenv import load_dotenv

# funzione per localizzare la root del progetto
def project_root() -> Path:
    """Restituisce la directory radice del progetto."""
    #a partire dal percorso corrente, sale e torna alla root del progetto
    return Path(__file__).resolve().parents[1]

def load_project_env() -> None:
    """
    Carica le variabili d'ambiente dal file .env situato nella radice del progetto.
    override=True permette di ricaricare le chiavi se vengono modificate al volo nel file .env
    """
    load_dotenv(project_root() / ".env", override=True)

#classe per la rotazione delle 2 chiavi API di Google, in modo da riuscire ad eseguire il sistema senza esaurire i token
class APIKeyRotator:
    def __init__(self):
        self.keys = [] #Lista che conterrà le chiavi trovate
        self.current_index = 0 #Indica l'indice la chiave utilizzata in quel momento

    def _load_keys_if_empty(self):
        """Carica le chiavi solo quando necessario, assicurando che il .env sia già attivo."""
        #Se la lista self.keys è vuota, perchè è la prima volta che viene invocata la funzione, inizia la ricerca a partire dall'indice 1 (gli indici sono solo 1 e 2)
        if not self.keys:
            idx = 1
            while True:
                key = os.getenv(f"GOOGLE_API_KEY_{idx}") #recupera la variabile d'ambiente GOOGLE_API_KEY_{idx}, che può essere GOOGLE_API_KEY_1 o GOOGLE_API_KEY_2
                if not key: #Se non c'è nessuna chiave, esce dal ciclo (uscirà comunque dopo 2 tentativi, visto che le chiavi sono 2)
                    break
                self.keys.append(key) #se trova la chiave, la aggiunge alla lista
                idx += 1
                
    #Metodo per alternare l'utilizzo delle chiavi con un algortimo Round Robin
    def get_next_key(self) -> str:
        # Assicuriamoci che le chiavi siano caricate DOPO che il .env è stato letto
        self._load_keys_if_empty()
        
        if not self.keys:
            raise RuntimeError("Nessuna chiave API Google trovata nel file .env! Verifica GOOGLE_API_KEY_1, ecc.")
            
        # Prende la chiave corrente
        key = self.keys[self.current_index]
        print(f"-> [KEY ROTATOR] Utilizzo la chiave API #{self.current_index + 1} di {len(self.keys)}")
        
        # Sposta l'indice sulla prossima chiave per la richiesta successiva
        self.current_index = (self.current_index + 1) % len(self.keys)
        return key

# Istanziamo il gestore come oggetto unico (Singleton)
key_rotator = APIKeyRotator()

#questa funzione viene invocata ad ogni chiamata all'LLM. Prima si caricano le variabili d'ambiente dal .env, poi si invoca il
#metodo get_next_key dell'istanza APIKeyRotator, che a sua volta prima carica le variabili d'ambiente (se non sono già state caricate)
#e poi alterna tra una chiave e l'altra 
def get_google_api_key() -> str:
    """Recupera la chiave API di Google verificandone la presenza (Nome preservato per compatibilità)."""
    load_project_env() 
    return key_rotator.get_next_key()