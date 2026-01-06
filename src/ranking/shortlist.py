def shortlist_candidates(final_scores, strong=0.6, borderline=0.5):
    shortlisted = []
    borderline_list = []
    rejected = []

    for name, scores in final_scores.items():
        score = scores["final_score"]

        if score >= strong:
            shortlisted.append((name, score))
        elif score >= borderline:
            borderline_list.append((name, score))
        else:
            rejected.append((name, score))

    return {
        "shortlisted": sorted(shortlisted, key=lambda x: x[1], reverse=True),
        "borderline": sorted(borderline_list, key=lambda x: x[1], reverse=True),
        "rejected": sorted(rejected, key=lambda x: x[1], reverse=True)
    }
