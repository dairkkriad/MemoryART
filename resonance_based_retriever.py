import os
import json
import re
import torch
from datetime import datetime
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# === Load local BERT ===
base_dir = os.path.dirname(os.path.abspath(__file__))
bert_path = os.path.join(base_dir, "bert")
_tokenizer = BertTokenizer.from_pretrained(bert_path)
_bert_model = BertModel.from_pretrained(bert_path)
_bert_model.eval()

@torch.no_grad()
def bert_encode(text: str):
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = _bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()

def safe_extract_date(date_str):
    try:
        parts = date_str.split()
        if len(parts) >= 2:
            return parts[1]
    except:
        pass
    return ""

def safe_extract_location(loc_str):
    try:
        return loc_str.split(":")[-1].strip()
    except:
        return ""

def parse_events_json(events_path):
    with open(events_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    all_events = []
    for record in raw:
        for ev in record.get("events", []):
            meta = ev.get("meta", {})
            parsed = {
                "date": safe_extract_date(meta.get("date", "")),
                "location": safe_extract_location(meta.get("location", "")),
                "diagnosis": ev.get("diagnosis", None),
                "symptoms": ev.get("symptoms", []),
                "treatment": ev.get("treatment", []),
                "medications": ev.get("medications", []),
                "lifestyle": ev.get("lifestyle", []),
                "people": ev.get("people", []),
                "follow_up": ev.get("follow_up", []),
                "summary": ev.get("summary", ""),
                "raw_dialogue": meta.get("raw_dialogue", []),
                "record_index": meta.get("record_index", -1), 
                "dialogue_index": meta.get("dialogue_index", -1),  
                "meta": meta 
            }
            all_events.append(parsed)
    return all_events

def parse_dataset_json(dataset_path):
    with open(dataset_path) as f:
        raw = json.load(f)

    qa_pairs = []
    for item in raw:
        for qa in item.get("QA_list", []):
            qa_pairs.append({"type": "qa", "question": qa["question"], "gold": qa.get("answer")})
        for mqa in item.get("mhop_QA_list", []):
            qa_pairs.append({"type": "mhop", "question": mqa["question"], "gold": mqa.get("answer")})
    return qa_pairs


def extract_question_clues(qtext: str):

    clues = {"date": None, "location": None, "disease": None, "between": [], "before": None, "has_disease": None}

    date_match = re.search(r'on\s+([A-Za-z]+\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})', qtext)
    if date_match:
        try:
            clues["date"] = str(datetime.strptime(date_match.group(1), "%B %d, %Y").date())
        except:
            clues["date"] = date_match.group(1)

    m = re.search(r'(contracted|recommend(?:ed)?)\s+(.*?)(?:\s+on|\?|\.|$)', qtext, re.IGNORECASE)
    if m:
        clues["disease"] = m.group(2).strip()

    if "where did" in qtext.lower():
        clues["location"] = True

    m_before = re.search(r'before contracting ([A-Za-z\s-]+)', qtext, re.IGNORECASE)
    if m_before:
        clues["before"] = m_before.group(1).strip()

    m_between = re.findall(r'contracting ([A-Za-z\s-]+)', qtext, re.IGNORECASE)
    if len(m_between) == 2:
        clues["between"] = [s.strip() for s in m_between]

    if "has the patient" in qtext.lower() or "ever contracted" in qtext.lower():
        m = re.search(r'(?:contracted|had)\s+([A-Za-z\s-]+)', qtext, re.IGNORECASE)
        if m:
            clues["has_disease"] = m.group(1).strip()

    return clues

def retrieve_matching_event(question: str, events: list):
    clues = extract_question_clues(question)
    best_match = None
    for ev in events:
        score = 0
        if clues["date"] and ev["date"] == clues["date"]:
            score += 3
        if clues["disease"]:
            d_lower = clues["disease"].lower()
            if ev["diagnosis"] and d_lower in ev["diagnosis"].lower():
                score += 2
            elif any(d_lower in s.lower() for s in ev["symptoms"]):
                score += 1
            elif any(d_lower in t.lower() for t in ev["treatment"]):
                score += 1
        if clues["location"] and ev["location"]:
            score += 1
        if best_match is None or score > best_match["score"]:
            best_match = {"event": ev, "score": score}
    return best_match

# if __name__ == "__main__":
#     events = parse_events_json("events.json")
#     qa_list = parse_dataset_json("dataset-episode-all.json")
#
#     print("\n多通道检索样例：")
#     for i, qa in enumerate(qa_list[:10]):
#         res = retrieve_matching_event(qa["question"], events)
#         print(f"[{qa['type']}] {qa['question']}")
#         print("→ 匹配得分:", res["score"], "| 匹配事件诊断:", res["event"].get("diagnosis"))
#         print("→ 日期:", res["event"].get("date"), "| 地点:", res["event"].get("location"))
#         print("-")
