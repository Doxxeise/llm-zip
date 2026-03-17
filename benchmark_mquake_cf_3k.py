"""
Continual Learning Atlas - REAL MQuAKE + RippleEdits Benchmark
PDC-Powered Automatic Entity Graph + ELQR Chaining

This script downloads the REAL MQuAKE-CF-3k benchmark from princeton-nlp/MQuAKE
and evaluates Atlas's multi-hop knowledge editing capability.

MQuAKE format:
  - requested_rewrite: list of cloze-style edits (subject, prompt, target_new)
  - questions: 3 multi-hop questions per case
  - new_answer: correct answer after editing
  - new_single_hops: single-hop QA after editing

We also include a RippleEdits-style evaluation for cascading consequences.
"""

import os
import re
import json
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANK = 64
LR = 5e-4
EPOCHS = 20
PDC_THRESHOLD = 2.0
SIMILARITY_THRESHOLD = 0.50
ELQR_THRESHOLD = 0.25
SAVE_DIR = "./atlas_real_mquake"
os.makedirs(SAVE_DIR, exist_ok=True)

# How many MQuAKE cases to evaluate (full=3000, but Colab T4 can do ~20-30)
MAX_CASES = 100  # Statistically meaningful sample (~30 min on T4)

print("=" * 65)
print(" CONTINUAL LEARNING ATLAS - REAL MQuAKE BENCHMARK")
print(" PDC-Powered Entity Graph + ELQR Multi-Hop Chaining")
print("=" * 65)


# ==========================================
# DOWNLOAD REAL MQuAKE DATASET
# ==========================================
def download_mquake():
    """Download MQuAKE-CF-3k from princeton-nlp GitHub."""
    import urllib.request
    url = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k.json"
    cache_path = os.path.join(SAVE_DIR, "MQuAKE-CF-3k.json")
    if os.path.exists(cache_path):
        print(f"  Found cached MQuAKE-CF-3k at {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)
    print(f"  Downloading MQuAKE-CF-3k from GitHub...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path, "r") as f:
            data = json.load(f)
        print(f"  Downloaded {len(data)} cases")
        return data
    except Exception as e:
        print(f"  Failed to download from GitHub: {e}")
        print(f"  Trying HuggingFace datasets...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Yiming/MQuAKE", "MQuAKE-CF-3k", split="test")
            data = [dict(row) for row in ds]
            with open(cache_path, "w") as f:
                json.dump(data, f)
            print(f"  Downloaded {len(data)} cases from HuggingFace")
            return data
        except Exception as e2:
            print(f"  HuggingFace also failed: {e2}")
            print(f"  Falling back to synthetic MQuAKE-style data...")
            return None


# ==========================================
# COMPONENTS (same as proven architecture)
# ==========================================
class EngramGatedChild(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.W_down = nn.Linear(d_model, rank, bias=False)
        self.W_up = nn.Linear(rank, d_model, bias=False)
        self.W_gate_down = nn.Linear(d_model, rank, bias=False)
        self.W_gate_up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.W_down.weight, std=0.01)
        nn.init.zeros_(self.W_up.weight)
        nn.init.normal_(self.W_gate_down.weight, std=0.01)
        nn.init.zeros_(self.W_gate_up.weight)

    def forward(self, h):
        value = self.W_up(F.gelu(self.W_down(h)))
        key = self.W_gate_up(F.gelu(self.W_gate_down(h)))
        gate = torch.sigmoid(
            (F.normalize(key, dim=-1) * F.normalize(h, dim=-1)).sum(dim=-1, keepdim=True)
        )
        return gate * value


def get_pdc_deviation_mask(base_model, input_ids, threshold=2.0, context_window=1):
    base_model.eval()
    with torch.no_grad():
        logits = base_model(input_ids).logits
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        raw_mask = (loss_per_token > threshold).float()
        expanded_mask = raw_mask.clone()
        seq_len = raw_mask.size(0)
        for i in range(seq_len):
            if raw_mask[i] == 1.0:
                start = max(0, i - context_window)
                end = min(seq_len, i + context_window + 1)
                expanded_mask[start:end] = 1.0
        final_mask = torch.cat([expanded_mask, torch.tensor([0.0]).to(DEVICE)])
        return loss_per_token, final_mask


def extract_pdc_entities(tokenizer, input_ids, mask):
    """Extract individual words from PDC-salient token spans."""
    token_ids = input_ids[0].tolist()
    mask_vals = mask.tolist()
    spans = []
    current_span = []
    for i, (tid, m) in enumerate(zip(token_ids, mask_vals)):
        if m > 0.5:
            current_span.append(tid)
        else:
            if current_span:
                spans.append(current_span)
                current_span = []
    if current_span:
        spans.append(current_span)

    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by', 'what', 'who', 'where', 'when',
                  'how', 'question', 'answer', 'that', 'this', 'it', 'and', 'or',
                  'does', 'did', 'do', 'above', 'all', 'others'}
    entity_words = set()
    for span in spans:
        text = tokenizer.decode(span, skip_special_tokens=True).strip()
        words = re.findall(r'[a-zA-Z]+', text.lower())
        for w in words:
            if len(w) > 1 and w not in stop_words:
                entity_words.add(w)
    return list(entity_words)


# ==========================================
# SPARSE ROUTING (BM25 + Stemming)
# ==========================================
def simple_stem(word):
    for suffix in ['tion', 'sion', 'ment', 'ness', 'ence', 'ance',
                   'ting', 'ing', 'ted', 'led', 'ied', 'ded',
                   'ful', 'ous', 'ive', 'ent', 'ant', 'ery',
                   'ator', 'tor', 'ter', 'ner', 'ler', 'der',
                   'or', 'er', 'ed', 'ly', 'es', 'al', 'en', 's']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def get_sparse_embedding(text):
    words = re.findall(r'\w+', text.lower())
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by', 'what', 'who', 'where', 'when',
                  'how', 'question', 'answer', 'that', 'this', 'it', 'does', 'did'}
    stemmed = [simple_stem(w) for w in words if w not in stop_words]
    return Counter(stemmed)


def compute_sparse_similarity(query_emb, doc_emb):
    if not query_emb or not doc_emb:
        return 0.0
    query_words = set(query_emb.keys())
    doc_words = set(doc_emb.keys())
    if not query_words:
        return 0.0
    intersection = query_words.intersection(doc_words)
    return len(intersection) / len(query_words)


# ==========================================
# GENERATION (Two-Pass)
# ==========================================
def generate_text(model, tokenizer, prompt, adapter=None, max_new_tokens=25):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            h_out = model(generated_ids, output_hidden_states=True).hidden_states[-1]
            if adapter is not None:
                h_float = h_out.float()
                base_norm = h_float.norm()
                h_float = h_float + adapter(h_float)
                h_out = (h_float * (base_norm / (h_float.norm() + 1e-8))).half()
            logits = model.lm_head(h_out)
            next_token_logits = logits[:, -1, :]

            generated_so_far = generated_ids[0, input_ids.shape[1]:].tolist()
            for token_id in set(generated_so_far):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= 1.3
                else:
                    next_token_logits[0, token_id] /= 1.3

            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    output_tokens = generated_ids[0, input_ids.shape[1]:]
    raw_output = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    if adapter is not None and raw_output:
        cleanup_prompt = f"Rewrite this with correct grammar: {raw_output}\nCorrected:"
        cleanup_ids = tokenizer.encode(cleanup_prompt, return_tensors="pt").to(DEVICE)
        cleanup_generated = cleanup_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(cleanup_generated)
                next_logits = outputs.logits[:, -1, :]
                next_tok = torch.argmax(next_logits, dim=-1).unsqueeze(-1)
                cleanup_generated = torch.cat([cleanup_generated, next_tok], dim=-1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break
        cleanup_tokens = cleanup_generated[0, cleanup_ids.shape[1]:]
        cleaned = tokenizer.decode(cleanup_tokens, skip_special_tokens=True).strip()
        return cleaned if cleaned else raw_output

    return raw_output


# ==========================================
# CONVERT MQuAKE CASE TO ATLAS FORMAT
# ==========================================
def mquake_case_to_facts(case):
    """
    Convert a real MQuAKE case into Atlas training facts.
    Each requested_rewrite becomes: "{subject} {relation} {target_new}"
    """
    facts = []
    rewrites = case.get("requested_rewrite", [])
    for i, rw in enumerate(rewrites):
        subject = rw.get("subject", "")
        prompt_template = rw.get("prompt", "{}")
        target_new = rw.get("target_new", {}).get("str", "")
        target_true = rw.get("target_true", {}).get("str", "")

        # Build the fact statement
        fact_text = prompt_template.replace("{}", subject) + " " + target_new + "."
        qa_text = f"Question: {rw.get('question', fact_text)} Answer: {target_new}."

        # Extract subject keywords from the subject entity
        subject_kw = [w.lower() for w in re.findall(r'[a-zA-Z]+', subject) if len(w) > 2]

        facts.append({
            "id": f"case{case.get('case_id', 0)}_edit{i}",
            "text": fact_text,
            "qa": qa_text,
            "subject_keywords": subject_kw,
            "target_new": target_new,
            "target_true": target_true,
            "subject": subject,
        })
    return facts


# ==========================================
# THE BENCHMARK
# ==========================================
def run_benchmark():
    # Download real MQuAKE
    print("\n  Downloading MQuAKE-CF-3k...")
    mquake_data = download_mquake()

    if mquake_data is None:
        print("  ERROR: Could not download MQuAKE. Exiting.")
        return

    # Filter to cases with 2+ hops for multi-hop testing
    multi_hop_cases = [c for c in mquake_data if len(c.get("requested_rewrite", [])) >= 2]
    print(f"  Total cases: {len(mquake_data)}, Multi-hop (2+ edits): {len(multi_hop_cases)}")

    # Sample subset for Colab
    test_cases = multi_hop_cases[:MAX_CASES]
    print(f"  Evaluating {len(test_cases)} cases on Colab T4\n")

    # Load model
    print(f"  Loading Base Model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    base_model.eval()
    d_model = base_model.config.hidden_size

    # =============================================
    print("\n" + "=" * 65)
    print("  PHASE 1: INGESTING ALL EDITS ACROSS ALL CASES")
    print("=" * 65)

    adapter_registry = {}
    entity_graph = {}
    centroid_index = {}

    # Collect all unique facts across all cases
    all_facts = []
    case_fact_map = {}  # case_id -> list of fact_ids
    for case in test_cases:
        case_id = case.get("case_id", 0)
        facts = mquake_case_to_facts(case)
        case_fact_map[case_id] = [f["id"] for f in facts]
        all_facts.extend(facts)

    print(f"\n  Total edits to ingest: {len(all_facts)}")
    start_time = time.time()

    for idx, fact in enumerate(all_facts):
        doc_id = fact["id"]
        texts = [fact["text"], fact["qa"]]

        adapter = EngramGatedChild(d_model, RANK).to(DEVICE)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
        all_output_entities = []

        for epoch in range(EPOCHS):
            for text in texts:
                optimizer.zero_grad()
                input_ids = tokenizer.encode(text, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)
                with torch.no_grad():
                    h_base = base_model(input_ids, output_hidden_states=True).hidden_states[-1].detach().float()
                _, mask = get_pdc_deviation_mask(base_model, input_ids, threshold=PDC_THRESHOLD)

                if epoch == 0:
                    entities = extract_pdc_entities(tokenizer, input_ids, mask)
                    all_output_entities.extend(entities)

                if mask.sum() == 0:
                    continue

                h_child = h_base + adapter(h_base)
                logits = base_model.lm_head(h_child.half()).float()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                shifted_mask_bool = mask[1:].bool()
                soft_mask = torch.where(shifted_mask_bool, torch.tensor(1.0, device=DEVICE), torch.tensor(0.1, device=DEVICE))
                loss = (losses * soft_mask).sum() / (soft_mask.sum() + 1e-9)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()

        adapter.eval()
        torch.save(adapter.state_dict(), os.path.join(SAVE_DIR, f"{doc_id}.pt"))
        adapter_registry[doc_id] = adapter

        output_entities = list(set(all_output_entities))
        input_keywords = fact.get("subject_keywords", [])
        sparse_emb = get_sparse_embedding(texts[0])

        entity_graph[doc_id] = {
            "input_entities": input_keywords,
            "output_entities": [e.lower() for e in output_entities],
            "sparse_emb": sparse_emb,
        }
        centroid_index[doc_id] = sparse_emb

        if (idx + 1) % 5 == 0:
            print(f"    Ingested {idx + 1}/{len(all_facts)} edits...")

        del optimizer
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/max(len(all_facts),1):.2f}s/edit)")

    # =============================================
    # Build Entity Graph Links (CASE-SCOPED)
    # =============================================
    print("\n" + "=" * 65)
    print("  PHASE 2: PDC ENTITY GRAPH (Case-Scoped)")
    print("=" * 65)

    link_count = 0
    for src_id, src_meta in entity_graph.items():
        src_case = src_id.split("_edit")[0]
        for dst_id, dst_meta in entity_graph.items():
            if src_id == dst_id:
                continue
            # CASE-SCOPING: only link adapters within the same edit group
            dst_case = dst_id.split("_edit")[0]
            if dst_case != src_case:
                continue
            src_outputs = set(src_meta["output_entities"])
            dst_inputs = set(dst_meta["input_entities"])
            overlap = src_outputs.intersection(dst_inputs)
            if overlap:
                link_count += 1
    print(f"  Entity graph: {len(entity_graph)} nodes, {link_count} case-scoped links")

    # =============================================
    # Retrieval and ELQR Functions
    # =============================================
    def retrieve_adapter(query_text, threshold=SIMILARITY_THRESHOLD):
        query_emb = get_sparse_embedding(query_text)
        best_id, best_sim = None, -1.0
        for doc_id, doc_emb in centroid_index.items():
            sim = compute_sparse_similarity(query_emb, doc_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = doc_id
        return best_id, best_sim

    def find_entity_links(source_id):
        """Find linked adapters WITHIN THE SAME CASE, ranked by specificity.
        Case-scoping prevents cross-case contamination at scale (O(k²) per case, not O(n²) global).
        """
        src_case = source_id.split("_edit")[0]
        src_outputs = set(entity_graph[source_id]["output_entities"])
        links = []
        for dst_id, dst_meta in entity_graph.items():
            if dst_id == source_id:
                continue
            # CASE-SCOPING: only chain within the same edit group
            dst_case = dst_id.split("_edit")[0]
            if dst_case != src_case:
                continue
            dst_inputs = set(dst_meta["input_entities"])
            overlap = src_outputs.intersection(dst_inputs)
            if overlap:
                links.append((dst_id, len(overlap), overlap))
        # Sort by specificity: most overlapping entities first
        links.sort(key=lambda x: x[1], reverse=True)
        return [(l[0], l[2]) for l in links]

    def elqr_multi_hop(query_text, max_hops=3):
        chain = []
        best_id, best_sim = retrieve_adapter(query_text)
        if best_sim < ELQR_THRESHOLD:
            return generate_text(base_model, tokenizer, query_text), chain

        chain.append(best_id)
        current_id = best_id

        for hop in range(2, max_hops + 1):
            linked = find_entity_links(current_id)
            if not linked:
                break
            next_id = linked[0][0]  # Most specific link
            chain.append(next_id)
            current_id = next_id

        final_adapter = adapter_registry[current_id]
        output = generate_text(base_model, tokenizer, query_text, adapter=final_adapter)
        return output, chain

    # =============================================
    print("\n" + "=" * 65)
    print("  PHASE 3: SINGLE-HOP EVALUATION")
    print("=" * 65)

    single_hop_correct = 0
    single_hop_total = 0

    for case in test_cases:
        case_id = case.get("case_id", 0)
        new_single_hops = case.get("new_single_hops", [])
        if not new_single_hops:
            continue

        for sh in new_single_hops:
            question = sh.get("question", "")
            cloze = sh.get("cloze", "")
            answer = sh.get("answer", "")
            aliases = sh.get("answer_alias", [])
            all_acceptable = [answer.lower()] + [a.lower() for a in aliases]

            query = cloze if cloze else question
            best_id, best_sim = retrieve_adapter(query)

            if best_sim >= SIMILARITY_THRESHOLD and best_id in adapter_registry:
                output = generate_text(base_model, tokenizer, query, adapter=adapter_registry[best_id])
            else:
                output = generate_text(base_model, tokenizer, query)

            match = any(ans in output.lower() for ans in all_acceptable)
            single_hop_correct += int(match)
            single_hop_total += 1

    single_acc = single_hop_correct / max(single_hop_total, 1) * 100
    print(f"  Single-hop: {single_hop_correct}/{single_hop_total} ({single_acc:.1f}%)")

    # =============================================
    print("\n" + "=" * 65)
    print("  PHASE 4: MULTI-HOP EVALUATION (PDC-ELQR)")
    print("=" * 65)

    multi_hop_correct = 0
    multi_hop_total = 0

    for case in test_cases:
        case_id = case.get("case_id", 0)
        questions = case.get("questions", [])
        new_answer = case.get("new_answer", "")
        new_answer_alias = case.get("new_answer_alias", [])
        all_acceptable = [new_answer.lower()] + [a.lower() for a in new_answer_alias]

        if not questions or not new_answer:
            continue

        case_pass = False
        for q in questions:
            output, chain = elqr_multi_hop(q)
            if any(ans in output.lower() for ans in all_acceptable):
                case_pass = True
                break

        if case_pass:
            multi_hop_correct += 1
        multi_hop_total += 1

        status = "✅" if case_pass else "❌"
        print(f"  {status} Case {case_id}: Expected '{new_answer}' | Got: '{output[:60]}' | Chain: {chain}")

    multi_acc = multi_hop_correct / max(multi_hop_total, 1) * 100

    # =============================================
    print("\n" + "=" * 65)
    print("  FINAL RESULTS — REAL MQuAKE-CF-3k BENCHMARK")
    print("=" * 65)

    print(f"""
  ┌──────────────────────────────┬──────────────┬──────────────┐
  │ Test                         │ Score        │ Accuracy     │
  ├──────────────────────────────┼──────────────┼──────────────┤
  │ Single-Hop (Edit Recall)     │ {single_hop_correct}/{single_hop_total}        │ {single_acc:.1f}%        │
  │ Multi-Hop (PDC-ELQR)         │ {multi_hop_correct}/{multi_hop_total}        │ {multi_acc:.1f}%        │
  └──────────────────────────────┴──────────────┴──────────────┘

  Entity Graph: {len(entity_graph)} nodes, {link_count} automatic links
  Chaining: PDC-Powered ELQR (Automatic Entity Linking)
  Model: {MODEL_NAME}
  Dataset: MQuAKE-CF-3k (first {MAX_CASES} multi-hop cases)
""")

    # Cleanup
    del base_model, adapter_registry
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_benchmark()
