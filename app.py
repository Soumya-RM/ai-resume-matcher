import streamlit as st
import os
import shutil
import json

from src.parsing.resume_parser import parse_all_resumes
from src.preprocessing.text_cleaner import preprocess_resume
from src.skill_extraction.rule_based import extract_skills_rule_based
from src.skill_extraction.semantic_extractor import extract_skills_semantic
from src.matching.matcher import compute_sentence_match,jd_coverage_score
from src.ranking.shortlist import shortlist_candidates


# ---------------- STREAMLIT CONFIG ---------------- #

st.set_page_config(
    page_title="AI Resume & Job Matcher",
    layout="wide"
)

st.title("📄 AI Resume & Job Matching System")
st.write("Upload resumes and match them against a job description using AI.")


# ---------------- PATHS ---------------- #

BASE_DIR = os.getcwd()
RESUME_DIR = os.path.join(BASE_DIR, "Data", "Resumes")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------- JOB DESCRIPTION ---------------- #

st.subheader("🧾 Job Description")

job_description = st.text_area(
    "Paste the job description here:",
    height=200
)

# ---------------- RESUME UPLOAD ---------------- #

st.subheader("📤 Upload Resumes (PDF / DOCX)")

uploaded_files = st.file_uploader(
    "Upload one or more resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    # Clear old resumes
    for f in os.listdir(RESUME_DIR):
        os.remove(os.path.join(RESUME_DIR, f))

    # Save uploaded resumes
    for file in uploaded_files:
        with open(os.path.join(RESUME_DIR, file.name), "wb") as f:
            f.write(file.read())

    st.success(f"{len(uploaded_files)} resume(s) uploaded successfully!")


# ---------------- RUN PIPELINE ---------------- #

if st.button("🚀 Run Resume Matcher"):

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not os.listdir(RESUME_DIR):
        st.error("Please upload at least one resume.")
        st.stop()

    st.info("Running AI matching pipeline...")

    # ---- Parsing ----
    resumes = parse_all_resumes(RESUME_DIR)

    # # ---- Skill Extraction ----
    # final_resume_skills = {}

    # for name, text in resumes.items():
    #     clean_text = preprocess_resume(text)

    #     final_resume_skills[name] = {
    #         "rule_based": extract_skills_rule_based(clean_text),
    #         "semantic": extract_skills_semantic(clean_text)
    #     }

    # ---- Matching ----
    job_skills = extract_skills_semantic(job_description)

    final_scores = {}

    for name, resume_text in resumes.items():

        # --- JD coverage ---
        coverage_score = jd_coverage_score(
            resume_text,
            job_skills
        )
        coverage_score = min(max(coverage_score, 0.0), 1.0)

        # --- Sentence similarity ---
        sentence_score = compute_sentence_match(
            resume_text,
            job_description
        )
        sentence_score = min(max(sentence_score, 0.0), 1.0)

        # --- Raw similarity ---
        raw_score = (
            0.75 * coverage_score +
            0.25 * sentence_score
        )

        # --- ATS-style normalization ---
        final_score = round(
            min(raw_score * 1.6 + 0.05, 1.0),
            3
        )

        final_scores[name] = final_score









    # ---- Shortlisting ----
    shortlist = shortlist_candidates(
        {k: {"final_score": v} for k, v in final_scores.items()}
    )

    # ---------------- DISPLAY RESULTS ---------------- #

    st.success("✅ Matching completed!")

    st.subheader("📊 Match Scores")
    st.json(final_scores)

    st.subheader("✅ Shortlisted Candidates")
    st.write(shortlist["shortlisted"])

    st.subheader("⚠️ Borderline Candidates")
    st.write(shortlist["borderline"])

    st.subheader("❌ Rejected Candidates")
    st.write(shortlist["rejected"])
