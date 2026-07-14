import os
import sys

# Allineamento dei path per caricare il VectorRAGManager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database.vector_manager import VectorRAGManager

def controlla_integrita_db():
    print(" Connessione a ChromaDB in corso...")
    rag_manager = VectorRAGManager()
    
    #Recuperiamo tutti i metadati presenti nel database vettoriale
    print(" Estrazione di tutti i documenti memorizzati...")
    all_data = rag_manager.vector_store.get(include=["metadatas"])
    
    if not all_data or not all_data.get("metadatas"):
        print(" Il database vettoriale è completamente vuoto!")
        return

    metadatas = all_data["metadatas"]
    print(f" Totale frammenti complessivi nel DB: {len(metadatas)}\n")
    
    # Contiamo quanti chunk ci sono per ogni singolo file sorgente
    conteggio_file = {}
    for meta in metadatas:
        if meta and "source" in meta:
            source_file = meta["source"]
            conteggio_file[source_file] = conteggio_file.get(source_file, 0) + 1
            
    #Stampiamo una tabella riassuntiva dei file indicizzati
    print(f"{'NOME FILE PDF':<70} | {'FRAMMENTI SALVATI':<15}")
    print("-" * 90)
    
    for file_name, numero_chunk in sorted(conteggio_file.items()):
        # Se un file ha pochissimi chunk (es. 1 o 2) ma è un file enorme, salterà subito all'occhio
        status_flag = " Parziale?" if numero_chunk <= 2 else " OK"
        print(f"{file_name:<70} | {numero_chunk:<5} ({status_flag})")

if __name__ == "__main__":
    controlla_integrita_db()