import os
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool("source_credibility_tool")
def source_credibility_tool(source_url_or_text: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Valuta l'affidabilità, l'autorevolezza e la qualità di una fonte web, di un URL
    o di un testo recuperato. Restituisce un punteggio di credibilità (0-100) 
    e un'analisi dei potenziali rischi (clickbait, fake news, mancanza di dati tecnici).
    """
    print(f"[TOOL INVOCATION] source_credibility_tool\n")
    print(f"Analisi per: '{source_url_or_text[:60]}...'")
    
    # Normalizza l'intero testo in minuscolo per evitare fallimenti di matching causati da maiuscole/minuscole
    text_to_analyze = source_url_or_text.lower()
    # Imposta la base di partenza del punteggio di credibilità (Ancoraggio Neutrale a 70)
    score = 70 
    # Inizializza la lista di stringhe che conterrà i log delle motivazioni del punteggio
    flags = []
    
    #Prima di tutto, effettuiamo un Analisi dei domini autorevoli (White-list editoriale)
    #Definiamo una Lista di domini di alto livello istituzionale o testate enciclopediche e specialistiche del settore
    authoritative = [
        "wikipedia", ".gov", ".edu", "soundonsound", "pitchfork", 
        "billboard", "rollingstone", "rockol", "ondarock", "instrumentalfx"
    ]
    if any(domain in text_to_analyze for domain in authoritative):
        score += 25
        flags.append("Dominio editoriale o istituzionale riconosciuto nel settore musicale.")
        
    #In secondo luogo, intercettiamo eventuali parole che potrbbero ricondurre ad un contenuto Clickbait o una Fake News
    #Definiamo una lista con le parole in questione
    clickbait_words = [
        "incredibile", "scioccante", "clicca qui", "segreto", "scandalo", 
        "assurdo", "shock", "clamoroso", "non crederai", "mistero", 
        "biglietti gratis", "last minute scioccante", "rivelato"
    ]
    found_clickbait = [w for w in clickbait_words if w in text_to_analyze]
    if found_clickbait:
        score -= 30
        flags.append(f"Rilevato linguaggio sensazionalistico/clickbait (parole: {found_clickbait}).")
        
    #Controlliamo la presenza di parole di dominio (Essenziale per il blog musicale)
    #Definiamo una lista di Parole chiave base che certificano che l'articolo parla effettivamente di musica
    musical_keywords = ["jazz", "rock", "pop", "canzoni", "canzone", "festival", "concerto", "live", "tour", "album", "band", "artista", "musica"]
    if any(keyword in text_to_analyze for keyword in musical_keywords):
        score += 5
        flags.append("La fonte contiene riferimenti semantici coerenti con il dominio musicale.")

    #Controlliamo la presenza di parole tecniche di dominio
    technical_terms = ["db", "hz", "khz", "gain", "impedenza", "risposta", "headroom", "vst", "fader", "analogico", "mixer"]
    live_terms = ["lineup", "biglietti", "ticket", "date", "location", "palco", "rassegna", "orari"]
    
    has_tech = any(term in text_to_analyze for term in technical_terms)
    has_live = any(term in text_to_analyze for term in live_terms)
    
    if has_tech or has_live:
        score += 10
        flags.append("La fonte contiene specifiche quantitative o dettagli organizzativi/tecnici appropriati.")
    else:
        score -= 15
    
        flags.append("La fonte tratta l'argomento in modo molto superficiale, priva di dettagli tecnici o specifiche di lineup/orari.")

    # Limitiamo il range tra 0 e 100
    score = max(0, min(100, score))
    
    
    if score >= 80:
        status = "ECCELLENTE (Altamente affidabile)"
    elif score >= 50:
        status = "ACCETTABILE (Usare con cautela)"
    else:
        status = "NON AFFIDABILE (Rischio elevato di fake news o allucinazione)"
    
    #Definiamo l'output da passare alla cronologia dei messaggi
    output = [
        "[OBSERVATION - VERIFICA DI CREDIBILITÀ DELLA FONTE]",
        f"TARGET ANALIZZATO: {source_url_or_text}", 
        f"PUNTEGGIO DI AFFIDABILITÀ: {score}/100",
        f"STATUS VALUTATIVO: {status}",
        "\nFattori riscontrati empiricamente durante l'analisi:"
    ]
    
    #Aggiungiamo i flag popolati in precedenza con la presenza di determinate caratteristiche all'output
    if flags:
        for f in flags:
            output.append(f"  - {f}")
    else:
        output.append("  - Nessun indicatore specifico. Fonte valutata come neutrale standard.")
        
    # Unisce le righe dell'output usando il carattere di andata a capo (\n)
    final_text_observation = "\n".join(output)
    
    return Command(
        update={
            "messages": [ToolMessage(content=final_text_observation, name="source_credibility_tool", tool_call_id=tool_call_id)]
        }
    )

credibility_tool = source_credibility_tool