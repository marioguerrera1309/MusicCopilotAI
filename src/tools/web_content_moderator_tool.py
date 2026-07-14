import re
import os
import sys
from langchain_core.tools import tool
from langgraph.types import Command # per modificare lo stato
from langchain_core.messages import ToolMessage # per scrivere un messaggio nella cronologia "messages"
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline # componenti di HugginFace per caricare modello e tokenizer
from typing import Annotated
from langchain_core.tools import InjectedToolCallId # per iniettare l'id della chiamata corrente del tool nella firma della funzione
import textwrap # per spezzare le righe chilometriche in frammenti più piccoli

# Inizializzazione della pipeline RoBERTa Full Fine-Tuned
MODEL_PATH = "./fine-tuning/roberta_full" # percorso dove è salvato il modello roBERTa full fine tuned
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH) # Carica il tokenizer salvato, configurato appositamente con le regole per l'italiano e max_length=512.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH) # Carica i pesi del modello addestrato per la classificazione sequenziale.

# RILEVAMENTO HARDWARE: Cerca una GPU NVIDIA (0) o Apple Silicon (mps), altrimenti ripiega su CPU (-1)
if torch.cuda.is_available():
    device = 0
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = -1

print(f"-> [INFO] Dispositivo di inferenza selezionato: {device}")

# Istanzia una pipeline di astrazione di Hugging Face per eseguire inferenze rapide sul testo.
# truncation=True garantisce che blocchi enormi vengano tagliati ai 512 token limite in automatico, senza mandare in crash la GPU
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, device=device) 

@tool # decoratore: trasforma la funzione in un tool utilizzabile dall'LLM
def web_content_moderator_tool(
    raw_observation_content: str, 
    target_tool_call_id: str, 
    query: str,
    # INIETTA L'ID DELLA CHIAMATA CORRENTE DEL MODERATORE
    moderator_tool_call_id: Annotated[str, InjectedToolCallId] 
) -> Command:
    """
    Analizza il contenuto testuale di una precedente ricerca web.
    Classifica paragrafo per paragrafo escludendo lo spam e il gossip pubblicitario.
    Sovrascrive chirurgicamente il vecchio ToolMessage mantenendo la formattazione originale.
    """
    print(f"\n--- [RoBERTa Guardrail] Avvio moderazione sul tool_call_id: {target_tool_call_id} ---")
    

    # Separazione dell'intestazione originale rispetto ai blocchi delle fonti
    # Isoliamo i singoli paragrafi o blocchi di testo delle fonti
    paragraphs = raw_observation_content.split("\n") 
    
    # Definiamo lo stesso formato della ricerca restituita da TAVILY nel web_search_tool
    header = [
        "[OBSERVATION]=== RISULTATI RICERCA WEB ESTERNA (TAVILY) ===",
        f"Evidenze pubbliche trovate in tempo reale su internet per la query '{query}':" 
    ]
    
    approved_blocks = [] # lista in cui verranno inseriti solo i paragrafi approvati dal modello
    
    # Scansione e classificazione
    current_block = [] 
    for line in paragraphs:
        # Se incontriamo un separatore di Tavily o l'inizio di una nuova fonte, elaboriamo il blocco accumulato fino a quel momento.
        if "---" in line or "Fonte (URL):" in line:
            if current_block: # Se il buffer non è vuoto, significa che c'è del testo da analizzare.
                block_text = "\n".join(current_block) 
                # Invocazione del modello
                prediction = classifier(block_text)[0] 
                print(f"-> [RoBERTa] Blocco di {len(block_text)} caratteri classificato come '{prediction['label']}' con confidenza {prediction['score']:.2f}")
                if prediction["label"] == "divulgativo":
                    approved_blocks.append(block_text) 
                else:
                    print(f"-> [SPAM FILTRATO] Rimosso blocco pubblicitario (Confidenza: {prediction['score']:.2f})")
                
                current_block = [] # Svuota il buffer per prepararlo ad ospitare la fonte successiva.
                
            if "Fonte (URL):" in line:
                current_block.append(line) # Preserva la riga dell'URL inserendola all'inizio del nuovo blocco.
        else:
            # Se la riga contiene del testo e non fa parte dell'intestazione fissa iniziale
            if line.strip() and not any(h in line for h in header):
                # Spezza la riga in frammenti di massimo 600 caratteri.
                # break_long_words=False fa in modo che gli spazi vengano rispettati e le parole non vengano divise a metà.
                # wrap() restituisce una lista di frammenti, ognuno con lunghezza <= 600 caratteri.
                frammenti = textwrap.wrap(line, width=600, break_long_words=False)

                # Aggiungiamo i frammenti al blocco uno per volta
                for frammento in frammenti:
                    current_block.append(frammento)
                    testo_accumulato = "\n".join(current_block)
                    # Appena il blocco raggiunge una dimensione critica (600), lo facciamo analizzare subito, senza aspettare la fine della fonte
                    if len(testo_accumulato) >= 600:
                        prediction = classifier(testo_accumulato)[0] 
                        print(f"-> [RoBERTa] Blocco di {len(testo_accumulato)} caratteri classificato come '{prediction['label']}' con confidenza {prediction['score']:.2f}")
                        if prediction["label"] == "divulgativo":
                            approved_blocks.append(testo_accumulato) 
                        else:
                            print(f"-> [SPAM FILTRATO] Rimosso blocco pubblicitario (Confidenza: {prediction['score']:.2f})")

                        # Svuota il buffer per far spazio ai prossimi frammenti della stessa pagina
                        current_block = []
                
    # Gestione dell'ultimo blocco rimasto fuori dal ciclo for (l'ultimo blocco di testo non seguito da un separatore ---).
    if current_block:
        block_text = "\n".join(current_block)
        prediction = classifier(block_text)[0]
        
        # Log visivi per l'ultimo blocco
        print(f"-> [RoBERTa] Blocco di {len(block_text)} caratteri classificato come '{prediction['label']}' con confidenza {prediction['score']:.2f}")
        
        if prediction["label"] == "divulgativo":
            approved_blocks.append(block_text)
        else:
            print(f"-> [SPAM FILTRATO] Rimosso blocco pubblicitario (Confidenza: {prediction['score']:.2f})")

    # Ricostruzione del testo finale unendo l'intestazione iniziale e le sole fonti approvate
    final_sources_text = "\n---\n".join(approved_blocks)
    final_text_observation = "\n".join(header) + "\n\n" + final_sources_text

    # Restituiamo l'oggetto COMMAND aggiornato con ENTRAMBI i messaggi
    return Command(
        update={
            "messages": [
                # Il messaggio che SOVRASCRIVE la vecchia ricerca di Tavily
                ToolMessage(
                    content=final_text_observation,
                    name="web_search_tool",
                    tool_call_id=target_tool_call_id, 
                    id=target_tool_call_id # Questo sovrascrive l'elemento nell'array dello stato globale
                ),
                # Il messaggio che CHIUDE la chiamata corrente dell'LLM
                ToolMessage(
                    content="[SISTEMA] Moderazione completata con successo. Contesto pulito.",
                    name="web_content_moderator_tool", # Nome di QUESTO tool
                    tool_call_id=moderator_tool_call_id # L'id iniettato in firma
                )
            ]
        }
    )