import os
import json

from src.parsing.resume_parser import parse_all_resumes
from src.preprocessing.text_cleaner import preprocess_resume
from src.skill_extraction.rule_based import extract_skills_rule_based
from src.skill_extraction.semantic_extractor import extract_skills_semantic
from src.matching.matcher import compute_sentence_match, skill_match_score
from src.ranking.shortlist import shortlist_candidates

# ---------------- CONFIG ---------------- #



RESUME_DIR = "Data/Resumes"
RESULTS_DIR = "results"

JOB_DESCRIPTION = """
We are hiring a Data Scientist with strong experience in Python,
Machine Learning, SQL, and data analysis. Hands-on experience with
Pandas, NumPy, Scikit-learn is required.
Knowledge of Deep Learning (CNN, RNN, LSTM, BERT),
TensorFlow and Keras is a plus.
"""

import os
os.makedirs(RESULTS_DIR, exist_ok=True)



# ---------------- MODULE 1: PARSING ---------------- #

print("🔹 Parsing resumes...")
resumes = parse_all_resumes(RESUME_DIR)

# ---------------- MODULE 2: SKILL EXTRACTION ---------------- #

print("🔹 Extracting skills...")
final_resume_skills = {}

for name, text in resumes.items():
    clean_text = preprocess_resume(text)

    rule_skills = extract_skills_rule_based(clean_text)
    semantic_skills = extract_skills_semantic(clean_text)

    final_resume_skills[name] = {
        "rule_based_skills": rule_skills,
        "semantic_bert_skills": semantic_skills
    }

with open(f"{RESULTS_DIR}/resume_skills_final.json", "w") as f:
    json.dump(final_resume_skills, f, indent=4)

# ---------------- MODULE 3: MATCHING ---------------- #

print("🔹 Matching resumes with job description...")
job_skills = extract_skills_semantic(JOB_DESCRIPTION)

final_scores = {}

for name, resume_text in resumes.items():
    resume_skills = final_resume_skills[name]["semantic_bert_skills"]

    skill_score = skill_match_score(resume_skills, job_skills)
    sentence_score = compute_sentence_match(resume_text, JOB_DESCRIPTION)

    final_score = round(0.7 * skill_score + 0.3 * sentence_score, 3)

    final_scores[name] = {
        "skill_score": skill_score,
        "sentence_score": sentence_score,
        "final_score": final_score
    }

with open(f"{RESULTS_DIR}/final_match_scores.json", "w") as f:
    json.dump(final_scores, f, indent=4)

# ---------------- MODULE 4: SHORTLISTING ---------------- #

print("🔹 Shortlisting candidates...")
shortlist_results = shortlist_candidates(final_scores)

with open(f"{RESULTS_DIR}/shortlisted_candidates.json", "w") as f:
    json.dump(shortlist_results, f, indent=4)

# ---------------- DONE ---------------- #

print("\n✅ Pipeline completed successfully!")
print("📁 Results saved in /results folder")
