import random, re, torch
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModelForCausalLM

_word_re = re.compile(r"^[A-Za-z][A-Za-z\-']+$")   # ignore punct, numbers

class PerturbationCurvature:
    def __init__(self, model_name="gpt2-medium", k_swaps=5, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.lm  = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.k   = k_swaps
        self.device = device

    def _synonym(self, word: str) -> str:
        """Return a random WordNet synonym; fallback to original word."""
        if not _word_re.match(word):
            return word
        lemmas = {
            l.name().replace('_', ' ')
            for syn in wn.synsets(word)
            for l in syn.lemmas()
            if l.name().lower() != word.lower()
        }
        return random.choice(list(lemmas)) if lemmas else word

    @torch.no_grad()
    def score(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        base_ids = self.tok(text, return_tensors="pt").to(self.device)
        base_lp  = self.lm(**base_ids, labels=base_ids.input_ids).loss.item()

        idxs   = random.sample(range(len(words)), min(self.k, len(words)))
        deltas = []
        for i in idxs:
            pert = words.copy()
            pert[i] = self._synonym(pert[i])
            pert_ids = self.tok(" ".join(pert), return_tensors="pt").to(self.device)
            pert_lp  = self.lm(**pert_ids, labels=pert_ids.input_ids).loss.item()
            deltas.append(pert_lp - base_lp)
        return sum(deltas) / len(deltas) if deltas else 0.0
