import os
import sys
from typing import Annotated #Per aggiungere metadati alle variabili
from langchain_core.tools import tool #decoratore per trasformare una funzione python in un tool per l'LLM
from langchain_core.tools import InjectedToolCallId #per iniettare l'ID univoco della chiamata del tool direttamente dal runtime di LangGraph
from langgraph.types import Command #Per modificare lo stato globale direttamente da un tool
from langchain_core.messages import ToolMessage

# Allineamento dei path per raggiungere i moduli interni
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.database.kg_manager import EditorialKGManager


def _to_msgpack_safe(value):
    """
    Normalizza neo4j.time.DateTime in stringhe ISO serializzabili da msgpack.
    """
    if isinstance(value, dict):
        return {k: _to_msgpack_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_msgpack_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Python datetime/date/time
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    # Neo4j DateTime utilizzano iso_format()
    if hasattr(value, "iso_format"):
        try:
            return value.iso_format()
        except Exception:
            pass

    return str(value)

@tool("query_knowledge_graph_tool") #nome assegnato al tool
def query_knowledge_graph_tool(topic: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Interroga il Knowledge Graph editoriale (Neo4j) per recuperare sia i fatti chiave (claims)
    sia i TESTI INTEGRALI di articoli già pubblicati in passato inerenti a questo topic.
    Serve all'agente nella fase di ricerca e drafting per garantire consistenza,
    evitare contraddizioni ed estrarre nodi correlati.
    """
    print("\nLettura dell'archivio storico degli articoli su Neo4j... ")
    
    kg_manager = EditorialKGManager()
    
    try:
        #ripuliamo il topic passato come parametro, rimuovendo eventuali virgolette o apici
        clean_topic = topic.strip().replace('"', '').replace("'", "")
        print(f"-> Invocazione KG Tool con query parziale su Neo4j: '{clean_topic}'")
        
        # Eseguiamo la query Cypher definita nel manager
        context = kg_manager.get_context_for_topic(clean_topic)
        safe_context = _to_msgpack_safe(context)
        
        if not safe_context["exists"]:
            message_content = f"Il topic '{clean_topic}' non è mai stato trattato nel Knowledge Graph. È un ottimo gap di copertura da colmare."
            
            # Usiamo Command per aggiornare lo stato: notifichiamo al writer che il topic è un GAP
            return Command(
                update={
                    "context_knowledge_graph": {"exists": False, "previous_claims": [], "previous_full_posts": []},
                    ## Iniettiamo nella cronologia dei messaggi il ToolMessage formattato
                    "messages": [ToolMessage(content=message_content, name="query_knowledge_graph_tool", tool_call_id=tool_call_id)]
                },
            )
            
        output_lines = [
            f"[OBSERVATION]:=== CONTESTO STORICO RITROVATO PER IL TOPIC: '{safe_context['matched_topic']}' ==="
        ]
        
        #Parsing dei claim logici
        if safe_context.get("previous_claims"):
            output_lines.append("\n🔹 FATTI E CLAIM VERIFICATI NEL GRAFO:")
            for claim in safe_context["previous_claims"]:
                output_lines.append(f"  - {claim}")
                
        # Iniezione dell'intero post d'archivio approvato
        if safe_context.get("previous_full_posts"):
            output_lines.append("\n" + "="*60)
            output_lines.append(" TESTO COMPLETO DEI POST PRECEDENTI APPROVATI IN ARCHIVIO:")
            output_lines.append("="*60)
            for i, post in enumerate(safe_context["previous_full_posts"], 1):
                output_lines.append(f"\n[Articolo d'archivio #{i} - Titolo: '{post['title']}' ({post['date']})]")
                output_lines.append(f"--- INIZIO TESTO ORIGINALE ---\n{post['text']}\n--- FINE TESTO ORIGINALE ---")
                
        # Unisce tutte le righe dell'array inserendo un carattere di "a capo" (\n) creando il blocco di testo unico
        final_text_observation = "\n".join(output_lines)
        

        # Restituiamo Command: LangGraph prenderà il dizionario in 'update' e lo fonderà con l'AgentState.
        #con Command aggiorniamo sia lo stato generale dell'agente, sia i messaggi che utilizza il researcher per il ragionamento. 
        return Command(
            update={
                "context_knowledge_graph": safe_context,
                "messages": [ToolMessage(content=final_text_observation, name="query_knowledge_graph_tool", tool_call_id=tool_call_id)]
            },
        )
        
    except Exception as e:
        error_msg = f"Errore durante l'esecuzione del tool Knowledge Graph: {str(e)}"
        return Command(
            update={
                "context_knowledge_graph": {"exists": False},
                "messages": [ToolMessage(content=error_msg, name="query_knowledge_graph_tool", tool_call_id=tool_call_id)]
            },
        )
        
    finally:
        kg_manager.close()