import os
import json
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from retriever import retrieve_by_clues_or_embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def build_prompt(memory_prompt: str, question: str) -> str:
    return f"""{memory_prompt}

Assume "now" refers to the most recent date in the memories above.

Answer the following question concisely, using only the memories above.

Formatting rules:
- Always express relative time (e.g., "yesterday", "last year", "next month") as full absolute dates like "7 May 2023" or "2022", using the most recent memory date as "now".
- Always use digits for durations (e.g., "4 years", not "four years").
- Use full and formal terms only (e.g., "transgender woman", not "trans woman").
- If multiple items apply, return a comma-separated list of short noun phrases. Do not use "or", "and", or sentence structures.
- For yes/no questions, respond only with "Yes" or "No".
- Do not use any reasoning, explanation, or restate the question.
- Never use complete sentences, linking verbs ("is", "was", "to"), or punctuation at the end.

Question: {question}
Answer:"""



DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_API_KEY="sk-6af1a5928c1b461291254272beda2792"

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.3
    }
    res = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"].strip()
    else:
        print("❌ API Error:", res.text)
        return ""


def compute_f1(pred_tokens, gold_tokens):
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def main():

    with open("qa_clues.json", "r", encoding="utf-8") as f:
        qa_clues = json.load(f)

    with open("locomo_person_memory.json", "r", encoding="utf-8") as f:
        memory_data = json.load(f)


    base_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_path = os.path.join(base_dir, "my_local_sbert")
    embed_model = HuggingFaceEmbedding(model_name=embedding_path)

    results = []
    smooth = SmoothingFunction().method4


    category_stats = {}

    for i, qa in enumerate(qa_clues):
        cat = str(qa["category"])  
        print(f"\n🧠 [{i+1}/{len(qa_clues)}] (Category {cat}) {qa['question']}")
        
        try:
            memory_prompt = retrieve_by_clues_or_embedding(qa, memory_data, embed_model)
            final_prompt = build_prompt(memory_prompt, qa["question"])


            est_tokens = len(final_prompt) // 4
            print(f"📦 Final prompt length: ~{est_tokens} tokens")

            answer = call_deepseek(final_prompt)

            ref = word_tokenize(str(qa["answer"]).lower())
            hyp = word_tokenize(str(answer).lower())
            
            bleu = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=smooth)
            f1 = compute_f1(hyp, ref)

            print(f"🤖 Model: {answer}")
            print(f"🎯 Gold : {qa['answer']}")
            print(f"✅ BLEU: {bleu:.4f} | F1: {f1:.4f}")


            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "bleu_sum": 0.0, "f1_sum": 0.0}
            category_stats[cat]["count"] += 1
            category_stats[cat]["bleu_sum"] += bleu
            category_stats[cat]["f1_sum"] += f1

            if category_stats[cat]["count"] % 10 == 0:
                avg_bleu = category_stats[cat]["bleu_sum"] / category_stats[cat]["count"]
                avg_f1 = category_stats[cat]["f1_sum"] / category_stats[cat]["count"]
                print(f"📊 [Category {cat}] has been processed {category_stats[cat]['count']} ")
                print(f"📈 Avg BLEU: {avg_bleu:.4f} | Avg F1: {avg_f1:.4f}")

            results.append({
                "question": qa["question"],
                "original_answer": qa["answer"],
                "conversation_id": qa["conversation_id"],
                "category": qa["category"],
                # "memory_prompt": memory_prompt,
                "model_answer": answer,
                "bleu": round(bleu, 4),
                "f1": round(f1, 4)
            })

            with open("qa_with_answers.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ Fail: {e}")

    print("\n📊 Result（final）：")

    for cat, stats in category_stats.items():
        count = stats["count"]
        if count == 0:
            continue
        avg_bleu = stats["bleu_sum"] / count
        avg_f1 = stats["f1_sum"] / count
        print(f"📂 Category {cat} → Sum: {count}")
        print(f"   🔹 Avg BLEU: {avg_bleu:.4f} | Avg F1: {avg_f1:.4f}")


if __name__ == "__main__":
    main()