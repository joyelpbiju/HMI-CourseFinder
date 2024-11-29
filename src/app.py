from flask import Flask, request, render_template, jsonify, send_file, abort
import sqlite3
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from googletrans import Translator
import re
import spacy
from mappings import synonym_mappings, keyword_mappings, number_words  # Import the mappings

app = Flask(__name__)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

translator = Translator()

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Absolute path to the database
db_path = 'E:/HSM/project_hmi/webappcoursespot/db/courses.sqlite'

# Define weights for each facet
FACET_WEIGHTS = {
    'title': 0.2,
    'instructor': 0.2,
    'term': 0.2,
    'credit': 0.1,
    'duration': 0.15,
    'type': 0.2,
    'lang': 0.2,
    'learning': 0.17,
    'content': 0.17,
}

# Columns to fetch directly if no facets are provided
DIRECT_COLUMNS = ["instructor", "title", "credits", "term", "medium_of_instruction"]


def parse_natural_language_query(query):
    doc = nlp(query)
    facets = {
        'instructor': None,
        'title': None,
        'term': None,
        'lang': None,
        'duration': None,
        'content': None,
        'learning': None,
        'credit': None,
        'type': None,
    }

    # Direct mappings for elective, obligatory, summer, and winter terms
    if "elective" in query.lower():
        facets['type'] = "elective"
    if "obligatory" in query.lower():
        facets['type'] = "obligatory"
    if "summer" in query.lower():
        facets['term'] = "summer"
    if "winter" in query.lower():
        facets['term'] = "winter"

    # Keyword-based extraction for OR condition
    for sentence in doc.sents:
        words = sentence.text.lower().split()
        if "or" in words:
            or_index = words.index("or")
            before_or = words[or_index - 1]
            after_or = words[or_index + 1]
            or_facet = None

            for facet, keywords in keyword_mappings.items():
                for keyword in keywords:
                    if keyword in sentence.text.lower():
                        or_facet = facet
                        break
                if or_facet:
                    break

            if or_facet:
                facets[or_facet] = f"{before_or}|{after_or}"

    # Further keyword-based extraction for additional context
    for sentence in doc.sents:
        words = sentence.text.lower().split()
        for facet, keywords in keyword_mappings.items():
            for keyword in keywords:
                if keyword in sentence.text.lower():
                    if keyword.split()[-1] in words:
                        keyword_index = words.index(keyword.split()[-1])

                        if facet == 'instructor':
                            next_word = words[keyword_index + 1] if keyword_index + 1 < len(words) else None
                            if next_word:
                                facets[facet] = next_word if not facets[facet] else f"{facets[facet]}|{next_word}"

                        elif facet == 'credit':
                            for i in range(keyword_index - 1, -1, -1):
                                if words[i].isdigit() and 1 <= int(words[i]) <= 10:
                                    facets[facet] = words[i] if not facets[facet] else f"{facets[facet]}|{words[i]}"
                                    break

                        elif facet == 'term' and not facets['term']:
                            if "summer" in query.lower():
                                facets[facet] = "summer"
                            elif "winter" in query.lower():
                                facets[facet] = "winter"
                            else:
                                if keyword in synonym_mappings:
                                    facets[facet] = synonym_mappings[keyword]
                                elif keyword_index + 1 < len(words):
                                    facets[facet] = words[keyword_index + 1]

                        elif facet == 'content':
                            keyword_indices = [i for i, word in enumerate(words) if word in keywords]
                            if len(keyword_indices) > 1:
                                keyword_index = keyword_indices[1]  # Use second keyword
                            next_word = doc[keyword_index + 1] if keyword_index + 1 < len(doc) else None
                            if next_word and next_word.pos_ == 'NOUN':
                                facets[facet] = next_word.text if not facets[
                                    facet] else f"{facets[facet]}|{next_word.text}"
                            else:
                                for token in doc[keyword_index + 1:]:
                                    if token.pos_ == 'NOUN':
                                        facets[facet] = token.text if not facets[
                                            facet] else f"{facets[facet]}|{token.text}"
                                        break

                        elif facet == 'lang':
                            if 'english' in query.lower():
                                facets[facet] = 'english'
                            elif 'german' in query.lower() or 'deutsch' in query.lower():
                                facets[facet] = 'german'
                            else:
                                if keyword in synonym_mappings:
                                    facets[facet] = synonym_mappings[keyword]
                                elif keyword_index + 1 < len(words):
                                    facets[facet] = words[keyword_index + 1]

                        elif facet == 'title':
                            next_word = words[keyword_index + 1] if keyword_index + 1 < len(words) else None
                            if next_word:
                                facets[facet] = next_word if not facets[facet] else f"{facets[facet]}|{next_word}"

                        elif facet == 'duration':
                            query_lower = query.lower()
                            if "1 semester" in query_lower:
                                facets[facet] = "1 semester"
                            elif "4 sws" in query_lower:
                                facets[facet] = "4 SWS"
                            else:
                                for word, num in number_words.items():
                                    if f"{word} semester" in query_lower:
                                        facets[facet] = f"{num} semester"
                                        break
                                else:
                                    for i in range(keyword_index + 1, len(words)):
                                        if words[i].isdigit() and 1 <= int(words[i]) <= 10:
                                            duration_str = f"{words[i]} {words[i + 1]}" if i + 1 < len(words) else \
                                            words[i]
                                            facets[facet] = duration_str if not facets[
                                                facet] else f"{facets[facet]}|{duration_str}"
                                            break

                        elif facet == 'learning':
                            keyword_indices = [i for i, word in enumerate(words) if word in keywords]
                            if len(keyword_indices) > 1:
                                keyword_index = keyword_indices[1]  # Use second keyword
                            next_word = doc[keyword_index + 1] if keyword_index + 1 < len(doc) else None
                            if next_word and next_word.pos_ == 'NOUN':
                                facets[facet] = next_word.text if not facets[
                                    facet] else f"{facets[facet]}|{next_word.text}"
                            else:
                                for token in doc[keyword_index + 1:]:
                                    if token.pos_ == 'NOUN':
                                        facets[facet] = token.text if not facets[
                                            facet] else f"{facets[facet]}|{token.text}"
                                        break

                        elif facet == 'type':
                            if "elective" in query.lower():
                                facets[facet] = "elective"
                            elif "obligatory" in query.lower() or "mandatory" in query.lower():
                                facets[facet] = "obligatory"
                            else:
                                if keyword in synonym_mappings:
                                    facets[facet] = synonym_mappings[keyword]
                                elif keyword_index + 1 < len(words):
                                    facets[facet] = words[keyword_index + 1]

    print("Parsed facets from natural language query:", facets)
    return {key: value for key, value in facets.items() if value is not None}


# Function to fetch data from the database
def fetch_data_from_db():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT rowid, title, instructor, time, term, duration, course_type, medium_of_instruction, credits, learning_obj, course_contents
            FROM merged_courses
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []


# Function to fetch course details from the database
def fetch_course_details(table_name, rowid):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT title, instructor, learning_obj, course_contents, teaching_methods, 
                   prerequisites, readings, applicability, workload, credits, evaluation, 
                   time, term, duration, course_type, medium_of_instruction, file_loc
            FROM {table_name} WHERE rowid = ?
        """, (rowid,))
        row = cursor.fetchone()
        conn.close()
        return row
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None


# Helper function to compute similarity
def compute_similarity(query_vector, text):
    text_vector = model.encode(text).tolist()
    return np.dot(query_vector, text_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(text_vector))


# Function to search courses based on multiple facets with weighted similarity for AND operator
def search_courses_by_facets(facets, threshold=0.3, penalty_factor=0.1):
    courses = []
    rows = fetch_data_from_db()

    for row in rows:
        rowid, title, instructor, time, term, duration, course_type, medium_of_instruction, credits, learning_obj, course_contents = row
        total_similarity = 0
        total_weight = 0
        matches = True

        for facet, query in facets.items():
            if facet == 'title':
                field_value = title
            elif facet == 'instructor':
                field_value = instructor
            elif facet == 'term':
                field_value = term
            elif facet == 'credit':
                field_value = credits
            elif facet == 'duration':
                field_value = duration
            elif facet == 'type':
                field_value = course_type
            elif facet == 'lang':
                field_value = medium_of_instruction
            elif facet == 'learning':
                field_value = learning_obj
            elif facet == 'content':
                field_value = course_contents
            else:
                continue

            if query is not None:
                facet_matches = False
                for sub_query in query.split('|'):
                    sub_query = sub_query.strip()
                    if field_value is not None:
                        if '*' in sub_query:
                            # Wildcard search
                            pattern = re.compile(sub_query.replace('*', '.*'), re.IGNORECASE)
                            if pattern.search(field_value):
                                facet_matches = True
                                similarity = 1.0
                                break
                        elif '"' in sub_query:
                            # Exact match search
                            if sub_query.replace('"', '').lower() in field_value.lower():
                                facet_matches = True
                                similarity = 1.0
                                break
                        else:
                            # Similarity search
                            query_vector = model.encode(sub_query).tolist()
                            similarity = compute_similarity(query_vector, field_value)
                            if similarity > threshold:
                                facet_matches = True
                                break

                if facet_matches:
                    total_similarity += similarity * FACET_WEIGHTS.get(facet, 0)
                    total_weight += FACET_WEIGHTS.get(facet, 0)
                else:
                    # Penalize if no match is found for this facet
                    total_similarity += penalty_factor * FACET_WEIGHTS.get(facet, 0)
                    total_weight += FACET_WEIGHTS.get(facet, 0)
                    matches = False
                    break

        if matches and total_weight > 0:
            weighted_similarity = total_similarity / total_weight
            courses.append({
                'table_name': 'merged_courses',
                'rowid': rowid,
                'title': title,
                'similarity': weighted_similarity,
                'instructor': instructor,
                'course_type': course_type,
                'term': term,
                'duration': duration,
                'medium_of_instruction': medium_of_instruction,
                'time': time,
                'credits': credits,
                'learning_obj': learning_obj,
                'course_contents': course_contents
            })

    return sorted(courses, key=lambda x: x['similarity'], reverse=True)


# Function to search courses based on a single facet
def search_courses_by_facet(facet, query, threshold=0.3):
    courses = []
    rows = fetch_data_from_db()

    for row in rows:
        rowid, title, instructor, time, term, duration, course_type, medium_of_instruction, credits, learning_obj, course_contents = row
        if facet == 'title':
            field_value = title
        elif facet == 'instructor':
            field_value = instructor
        elif facet == 'term':
            field_value = term
        elif facet == 'credit':
            field_value = credits
        elif facet == 'duration':
            field_value = duration
        elif facet == 'type':
            field_value = course_type
        elif facet == 'lang':
            field_value = medium_of_instruction
        elif facet == 'learning':
            field_value = learning_obj
        elif facet == 'content':
            field_value = course_contents
        else:
            continue

        facet_matches = False
        for sub_query in query.split('|'):
            sub_query = sub_query.strip()
            if field_value is not None:
                if '*' in sub_query:
                    # Wildcard search
                    pattern = re.compile(sub_query.replace('*', '.*'), re.IGNORECASE)
                    if pattern.search(field_value):
                        facet_matches = True
                        similarity = 1.0
                        break
                elif '"' in sub_query:
                    # Exact match search
                    if sub_query.replace('"', '').lower() in field_value.lower():
                        facet_matches = True
                        similarity = 1.0
                        break
                else:
                    # Similarity search
                    query_vector = model.encode(sub_query).tolist()
                    similarity = compute_similarity(query_vector, field_value)
                    if similarity > threshold:
                        facet_matches = True
                        break

        if facet_matches:
            courses.append({
                'table_name': 'merged_courses',
                'rowid': rowid,
                'title': title,
                'similarity': similarity,
                'instructor': instructor,
                'course_type': course_type,
                'term': term,
                'duration': duration,
                'medium_of_instruction': medium_of_instruction,
                'time': time,
                'credits': credits,
                'learning_obj': learning_obj,
                'course_contents': course_contents
            })

    return sorted(courses, key=lambda x: x['similarity'], reverse=True)


# Function to search courses based on direct column matches
def search_courses_direct(query):
    courses = []
    rows = fetch_data_from_db()
    query_vector = model.encode(query).tolist()

    for row in rows:
        rowid, title, instructor, time, term, duration, course_type, medium_of_instruction, credits, learning_obj, course_contents = row
        matches = []

        for column_name, field_value in zip(DIRECT_COLUMNS, [instructor, title, credits, term, medium_of_instruction]):
            if field_value is not None:
                similarity = compute_similarity(query_vector, field_value)
                matches.append((similarity, column_name, field_value))

        # Get the best match
        best_match = max(matches, key=lambda x: x[0], default=None)

        if best_match and best_match[0] > 0.3:  # Threshold for considering a match
            courses.append({
                'table_name': 'merged_courses',
                'rowid': rowid,
                'title': title,
                'similarity': best_match[0],
                'instructor': instructor,
                'course_type': course_type,
                'term': term,
                'duration': duration,
                'medium_of_instruction': medium_of_instruction,
                'time': time,
                'credits': credits,
                'learning_obj': learning_obj,
                'course_contents': course_contents
            })

    return sorted(courses, key=lambda x: x['similarity'], reverse=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    courses = []

    if not query:
        return jsonify({'courses': []})

    if any(facet in query for facet in
           ['instructor:', 'title:', 'term:', 'lang:', 'duration:', 'content:', 'learning:', 'credit:', 'type:']):
        facets = {}
        for part in query.split(','):
            if ':' in part:
                key, value = part.split(':', 1)
                facets[key.strip()] = value.strip()
        print("Explicit query facets:", facets)
    else:
        if '*' in query or '"' in query:
            # Wildcard or exact match search
            facets = {'wildcard_or_exact': query}
        else:
            facets = parse_natural_language_query(query)
            print("Implicit query converted to explicit facets:", facets)

    if facets:
        if 'wildcard_or_exact' in facets:
            query = facets['wildcard_or_exact']
            courses = search_courses_by_facet('wildcard_or_exact', query, threshold=0.3)
        elif len(facets) > 1:
            courses = search_courses_by_facets(facets, threshold=0.3)
        else:
            facet = list(facets.keys())[0]
            query = facets[facet]
            courses = search_courses_by_facet(facet, query, threshold=0.3)
    else:
        print("No facets found in the query, performing direct column search.")
        courses = search_courses_direct(query)

    print(f"Found {len(courses)} courses matching the query.")
    return jsonify({'courses': courses})


@app.route('/course/<table_name>/<int:rowid>')
def course_details(table_name, rowid):
    course_details = fetch_course_details(table_name, rowid)
    if not course_details:
        return jsonify({'error': 'Course not found'}), 404

    return jsonify({
        'title': course_details[0],
        'instructor': course_details[1],
        'learning_obj': course_details[2],
        'course_contents': course_details[3],
        'teaching_methods': course_details[4],
        'prerequisites': course_details[5],
        'readings': course_details[6],
        'applicability': course_details[7],
        'workload': course_details[8],
        'credits': course_details[9],
        'evaluation': course_details[10],
        'time': course_details[11],  # Not translated
        'term': course_details[12],
        'duration': course_details[13],
        'course_type': course_details[14],
        'medium_of_instruction': course_details[15],  # Not translated
        'file_loc': course_details[16]  # Not translated
    })


@app.route('/files/<path:filepath>')
def download_file(filepath):
    try:
        # Verify the file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        print(f"Error serving file: {e}")
        abort(500, description="Error serving file")


@app.route('/translate/<table_name>/<int:rowid>/<lang>', methods=['GET'])
def translate_course_details(table_name, rowid, lang):
    course_details = fetch_course_details(table_name, rowid)
    if not course_details:
        return jsonify({'error': 'Course not found'}), 404

    translated_details = []
    try:
        for text in course_details[
                    :-2]:  # Translate all except the last two fields (file_loc and medium_of_instruction)
            if text:
                if lang == 'en':
                    translated_text = translator.translate(text, src='de', dest='en').text
                else:
                    translated_text = translator.translate(text, src='en', dest='de').text
                translated_details.append(translated_text)
            else:
                translated_details.append(text)
    except Exception as e:
        print(f"Translation error: {e}")
        translated_details = ["Translation error" if text else text for text in course_details[:-2]]

    # Add back the last two fields without translation
    translated_details.append(course_details[-2])  # file_loc
    translated_details.append(course_details[-1])  # medium_of_instruction

    return jsonify({
        'table_name': table_name,
        'rowid': rowid,
        'title': translated_details[0],
        'instructor': translated_details[1],
        'learning_obj': translated_details[2],
        'course_contents': translated_details[3],
        'teaching_methods': translated_details[4],
        'prerequisites': translated_details[5],
        'readings': translated_details[6],
        'applicability': translated_details[7],
        'workload': translated_details[8],
        'credits': translated_details[9],
        'evaluation': translated_details[10],
        'time': course_details[11],  # Not translated
        'term': translated_details[12],
        'duration': translated_details[13],
        'course_type': translated_details[14],
        'medium_of_instruction': course_details[15],  # Not translated
        'file_loc': course_details[16]  # Not translated
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
