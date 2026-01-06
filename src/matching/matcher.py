import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
# model = SentenceTransformer("all-MiniLM-L6-v2")


def split_into_sentences(text):
    text = re.sub(r"\n+", ".", text)
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    return sentences


def compute_sentence_match(resume_text, job_text):
    resume_sentences = split_into_sentences(resume_text)
    if not resume_sentences:
        return 0.0

    resume_embeddings = model.encode(resume_sentences, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)

    similarities = util.cos_sim(job_embedding, resume_embeddings)[0]
    return round(float(similarities.max()), 3)


def skill_match_score(resume_skills, job_skills):
    if not resume_skills or not job_skills:
        return 0.0

    resume_emb = model.encode(resume_skills, convert_to_tensor=True)
    job_emb = model.encode(job_skills, convert_to_tensor=True)

    similarity_matrix = util.cos_sim(job_emb, resume_emb)

    # For each job skill, take the best matching resume skill
    best_matches = similarity_matrix.max(dim=1).values

    return round(float(best_matches.mean()), 3)

def jd_coverage_score(resume_text, job_skills):
    if not resume_text.strip() or not job_skills:
        return 0.0

    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_skill_embeddings = model.encode(job_skills, convert_to_tensor=True)

    similarities = util.cos_sim(job_skill_embeddings, resume_embedding)

    raw_coverage = similarities.max(dim=1).values.mean()
    coverage_score = min(raw_coverage * 2.5, 1.0)

    return round(float(coverage_score), 3)


