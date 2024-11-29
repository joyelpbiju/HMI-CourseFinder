from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the fine-tuned model
model = SentenceTransformer('E:/HSM/project_hmi/webappcoursespot/fine tuned model')

def map_instructor_name(query, instructor_names):
    query_lower = query.lower()
    facets = {}

    # Check for exact matches
    for name in instructor_names:
        if name.lower() in query_lower:
            facets['instructor'] = query  # Use the original query part
            return facets

    # Check for partial matches using SBERT
    query_embedding = model.encode(query)
    max_similarity = 0
    best_match_start = None
    best_match_end = None
    for name in instructor_names:
        name_embedding = model.encode(name)
        similarity = np.dot(query_embedding, name_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(name_embedding))
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_start = query_lower.find(name.split()[0].lower())  # Get the start index of the partial match
            best_match_end = best_match_start + len(name.split()[0])  # Get the end index of the partial match

    if max_similarity > 0.3:  # Threshold for instructor similarity
        facets['instructor'] = query[best_match_start:best_match_end]

    return facets
