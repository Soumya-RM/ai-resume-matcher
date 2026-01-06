from sentence_transformers import SentenceTransformer, util
from src.skill_extraction.skill_vocab import SKILL_VOCAB
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def split_into_sentences(text):
    text = re.sub(r"\n+", ".", text)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    return sentences


def extract_skills_semantic(text, threshold=0.35):
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    skill_embeddings = model.encode(SKILL_VOCAB, convert_to_tensor=True)

    matched_skills = set()

    for idx, skill in enumerate(SKILL_VOCAB):
        similarities = util.cos_sim(skill_embeddings[idx], sentence_embeddings)[0]
        if similarities.max() >= threshold:
            matched_skills.add(skill)

    return sorted(matched_skills)
