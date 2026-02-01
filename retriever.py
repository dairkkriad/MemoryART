from difflib import SequenceMatcher
import numpy as np
import re
def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def similar_name(a: str, b: str, threshold=0.6) -> bool:
    return similarity(a, b) >= threshold

def extract_main_name(raw: str) -> str:
    NON_HUMAN = {"grandma", "daughter", "kid", "friend", "family", "husband", "wife", "pet", "dog", "cat", "mom", "dad", "son"}
    raw = raw.strip()
    raw = re.sub(r"'s\b", "", raw)
    raw = re.sub(r"[^a-zA-Z ,&]", "", raw)
    tokens = re.split(r"[ ,&]+", raw)
    for token in tokens:
        token = token.lower()
        if token and token not in NON_HUMAN:
            return token.capitalize()
    return ""

def is_empty_clue_set(clues: dict) -> bool:
    fields = ["person", "events", "relationships", "status_changes", "motivations"]
    return all(not clues.get(f, "").strip() for f in fields if f != "person")

def token_overlap_score(clue_str: str, mem_str: str) -> float:

    clue_tokens = set(clue_str.lower().split())
    mem_tokens = set(mem_str.lower().split())

    if not clue_tokens:
        return 0.0

    matched = clue_tokens & mem_tokens
    return len(matched) / len(clue_tokens)
def retrieve_by_clues_or_embedding(qa_clue, memory_data, embed_model, top_k=3) -> str:
    clues = qa_clue["clues"]
    conv_id = qa_clue["conversation_id"]
    question = qa_clue["question"]
    person_raw = clues.get("person", "")
    person_main = extract_main_name(person_raw)


    person_candidates = [
        m for m in memory_data
        if m["conversation_id"] == conv_id and similar_name(m["person"], person_main)
    ]


    all_clues_empty = is_empty_clue_set(clues)
    has_person = bool(person_main)
    has_person_match = len(person_candidates) > 0

    if (not has_person_match and all_clues_empty) or (not has_person and all_clues_empty):
        top_k = 20
    elif has_person_match and all_clues_empty:
        top_k = 20
    elif has_person_match and not all_clues_empty:
        top_k = 10

    prompt = "The following memories may help answer the question:\n\n"

    if (not has_person_match and all_clues_empty) or (not has_person and all_clues_empty):

        q_embed = embed_model.get_text_embedding(question)
        scored = []

        for m in memory_data:
            sm = m["structured_memory"]
            content = " ".join([
                m.get("person", ""),
                sm.get("summary", ""),
                " ".join(sm.get("events", [])),
                " ".join(sm.get("motivations", [])),
                " ".join(sm.get("status_changes", [])),
                " ".join(sm.get("relationships", [])),
            ])
            mem_embed = embed_model.get_text_embedding(content)
            score = np.dot(q_embed, mem_embed) / (np.linalg.norm(q_embed) * np.linalg.norm(mem_embed) + 1e-8)
            scored.append((score, m))

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]


    elif has_person_match and all_clues_empty:

        q_embed = embed_model.get_text_embedding(question)
        scored = []

        weights = {
            "summary": 0.1,
            "events": 0.5,
            "motivations": 0.25,
            "status_changes": 0.15,
            "relationships": 0.1
        }

        for m in person_candidates:
            sm = m["structured_memory"]
            total_score = 0

            for field, weight in weights.items():
                content = sm.get(field, "")
                if isinstance(content, list):
                    content = " ".join(content)
                if not content.strip():
                    continue

                mem_embed = embed_model.get_text_embedding(content)
                score = np.dot(q_embed, mem_embed) / (np.linalg.norm(q_embed) * np.linalg.norm(mem_embed) + 1e-8)
                total_score += score * weight

            scored.append((total_score, m))

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]

    else:

        scored = []
        weights = {
            "events": 0.7,
            "motivations": 0.25,
            "status_changes": 0.2,
            "relationships": 0.15
        }

        for m in person_candidates:
            sm = m["structured_memory"]
            total_score = 0
            used_weight = 0

            for field, weight in weights.items():
                clue_val = clues.get(field, "").strip()
                if not clue_val:
                    continue  
                mem_val = sm.get(field, [])
                mem_text = " ".join(mem_val) if isinstance(mem_val, list) else str(mem_val)
                score = token_overlap_score(clue_val, mem_text)
                total_score += score * weight
                used_weight += weight

            if used_weight > 0:
                total_score /= used_weight  
            else:
                total_score = 0

            scored.append((total_score, m))


        top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]


    prompt = "The following memories may help answer the question:\n\n"

    for score, mem in top:
        sm = mem["structured_memory"]
        utterances = mem.get("original_utterances", "").strip()
        date_time = mem.get("session_date_time", "")
        prompt += (
            f"Date: {date_time}\n"
            f"Original Utterances:\n{utterances}\n\n"
            f"Structured Memory:\n"
            f"  Events: {', '.join(sm.get('events', []))}\n"
            f"  Motivations: {', '.join(sm.get('motivations', []))}\n"
            f"  Status Changes: {', '.join(sm.get('status_changes', []))}\n"
            f"  Relationships: {', '.join(sm.get('relationships', []))}\n"
            f"  Summary: {sm.get('summary', '')}\n\n"
        )

    return prompt


