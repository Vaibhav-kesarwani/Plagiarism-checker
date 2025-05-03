import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set your folder path here
FOLDER_PATH = "/Users/vaibhavkesarwani/Desktop/Quiz_game/Docs"

# Read all .txt files from the specified folder
student_files = [doc for doc in os.listdir(FOLDER_PATH) if doc.endswith('.txt')]
student_notes = [open(os.path.join(FOLDER_PATH, _file), encoding='utf-8').read()
                 for _file in student_files]


def vectorize(Text): 
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity(doc1, doc2): 
    return cosine_similarity([doc1, doc2])


vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()


def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            percentage = round(sim_score * 100, 2)  # Convert to percentage
            student_pair = tuple(sorted((student_a, student_b)))
            score = (student_pair[0], student_pair[1], percentage)
            plagiarism_results.add(score)
    return plagiarism_results


# Print results
for student_a, student_b, match_percentage in check_plagiarism():
    print(f"{student_a} <--> {student_b} => {match_percentage}% match")
