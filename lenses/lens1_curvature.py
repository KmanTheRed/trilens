import random
import re
import torch
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForCausalLM

_word_re = re.compile(r"^[A-Za-z][A-Za-z\-']+$")   # ignore punctuation/nums

class PerturbationCurvature:
    def __init__(self,
                 model_name: str = "gpt2-medium",
                 k_swaps: int = 5,
                 device: str = "cuda"):
        # 1) tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # ensure we have a pad token for full‐length padding
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        # 2) model in FP16, low CPU RAM footprint
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.lm.eval()

        self.k      = k_swaps
        self.device = device

    def _synonym(self, word: str) -> str:
        """Return a random WordNet synonym; fallback to original."""
        if not _word_re.match(word):
            return word
        lemmas = {
            l.name().replace('_', ' ')
            for syn in wn.synsets(word)
            for l   in syn.lemmas()
            if l.name().lower() != word.lower()
        }
        return random.choice(list(lemmas)) if lemmas else word

    @torch.no_grad()
    def score(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0

        # --- base log‐prob
        base_ids = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.tok.model_max_length,
            padding="max_length"
        ).to(self.device)
        base_loss = self.lm(
            **base_ids,
            labels=base_ids.input_ids
        ).loss.item()

        # --- k perturbed versions
        idxs   = random.sample(range(len(words)), min(self.k, len(words)))
        deltas = []
        for i in idxs:
            pert = words.copy()
            pert[i] = self._synonym(pert[i])
            pert_ids = self.tok(
                " ".join(pert),
                return_tensors="pt",
                truncation=True,
                max_length=self.tok.model_max_length,
                padding="max_length"
            ).to(self.device)
            pert_loss = self.lm(
                **pert_ids,
                labels=pert_ids.input_ids
            ).loss.item()
            deltas.append(pert_loss - base_loss)

        return sum(deltas) / len(deltas)
