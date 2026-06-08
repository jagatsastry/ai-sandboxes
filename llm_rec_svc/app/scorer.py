"""LLM-based pointwise reranker.

Idea: for each (user_profile, item) we ask a small causal LM to score how
relevant the item is. We turn that into a number by reading the model's
log-prob of the token "yes" vs "no" after a fixed prompt. This is fast
(single forward pass, no generation loop) and works with any small HF
causal LM.

Latency tricks:
    - one HF model loaded at startup, moved to CUDA if available
    - fp16 on GPU via autocast
    - dynamic batching: all candidates scored in ONE forward pass
    - torch.inference_mode() everywhere
    - tokenizer padded to longest-in-batch only
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TMPL = (
    "You are a recommender. A user has this profile:\n"
    "{profile}\n\n"
    "Candidate item:\n"
    "Title: {title}\n"
    "Tags: {tags}\n"
    "Description: {desc}\n\n"
    "Is this item highly relevant to the user? Answer yes or no.\n"
    "Answer:"
)


@dataclass
class ScoreResult:
    item_id: str
    llm_score: float  # log p(yes) - log p(no), higher = more relevant
    retrieval_score: float  # cosine from candidate gen
    final_score: float  # weighted blend


class LLMScorer:
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2") -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.dtype).to(
            self.device
        )
        self.model.eval()

        # cache token ids for "yes"/"no" (leading space variants too)
        self._yes_ids = self._first_token_ids([" yes", "yes", " Yes", "Yes"])
        self._no_ids = self._first_token_ids([" no", "no", " No", "No"])

    def _first_token_ids(self, words: Sequence[str]) -> list[int]:
        ids = set()
        for w in words:
            t = self.tokenizer.encode(w, add_special_tokens=False)
            if t:
                ids.add(t[0])
        return sorted(ids)

    def warmup(self) -> None:
        dummy = [("profile", "title", ["tag"], "desc")]
        self.score_batch(dummy)

    @torch.inference_mode()
    def score_batch(
        self,
        rows: Sequence[tuple[str, str, Sequence[str], str]],
    ) -> tuple[list[float], dict]:
        """rows: list of (profile, title, tags, desc). Returns (scores, timings)."""
        if not rows:
            return [], {"tokenize_ms": 0.0, "gpu_ms": 0.0}

        prompts = [
            PROMPT_TMPL.format(profile=p, title=t, tags=", ".join(tags) or "(none)", desc=d)
            for (p, t, tags, d) in rows
        ]

        t0 = time.perf_counter()
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)
        t1 = time.perf_counter()

        if self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(**enc).logits
            torch.cuda.synchronize()
        else:
            logits = self.model(**enc).logits
        t2 = time.perf_counter()

        # Score = logit("yes") - logit("no") at the position of the LAST
        # real (non-pad) token of each prompt.
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1  # (B,)
        gathered = logits[torch.arange(logits.size(0)), last_idx]  # (B, V)
        log_probs = torch.log_softmax(gathered.float(), dim=-1)
        yes_lp = log_probs[:, self._yes_ids].logsumexp(dim=-1)
        no_lp = log_probs[:, self._no_ids].logsumexp(dim=-1)
        scores = (yes_lp - no_lp).cpu().tolist()

        return scores, {
            "tokenize_ms": (t1 - t0) * 1000.0,
            "gpu_ms": (t2 - t1) * 1000.0,
        }
