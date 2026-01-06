import re

SKILL_KEYWORDS = [
    "python", "machine learning", "deep learning", "sql", "tensorflow",
    "keras", "pandas", "numpy", "scikit-learn", "bert", "gpt",
    "lstm", "cnn", "rnn", "nlp"
]

def extract_skills_rule_based(text):
    text = text.lower()
    found_skills = set()

    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found_skills.add(skill)

    return list(found_skills)
