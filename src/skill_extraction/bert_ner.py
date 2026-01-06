from transformers import pipeline

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

TECH_ENTITIES = {"ORG", "MISC"}

def extract_skills_bert(text):
    entities = ner_pipeline(text)

    skills = set()
    for ent in entities:
        if ent["entity_group"] in TECH_ENTITIES:
            skills.add(ent["word"])

    return list(skills)
