from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import firebase_admin
from firebase_admin import credentials, firestore
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Firebase setup
if not firebase_admin._apps:
    firebase_creds_json = os.environ.get('FIREBASE_CREDENTIALS')
    if firebase_creds_json:
        try:
            creds_dict = json.loads(firebase_creds_json)
            if 'private_key' in creds_dict:
                creds_dict['private_key'] = creds_dict['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(creds_dict)
        except Exception as e:
            print(f"Error parsing FIREBASE_CREDENTIALS env var: {e}. Falling back to file.")
            cred = credentials.Certificate('firebase_key.json')
    else:
        cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Groq setup
client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Load SVD factors
with open('problem_factors.json', 'r') as f:
    problem_factors = json.load(f)

with open('student_factors.json', 'r') as f:
    student_factors = json.load(f)

# Load question bank
question_bank = pd.read_csv('question_bank.csv')

# ── Functions ──────────────────────────────────────

def build_student_factors(attempts):
    student_vec = np.random.normal(0, 0.1, 50)
    lr = 0.01
    for attempt in attempts:
        problem_id = str(attempt['problem_id'])
        is_correct = attempt['is_correct']
        if problem_id not in problem_factors:
            continue
        problem_vec = np.array(problem_factors[problem_id])
        predicted = np.dot(student_vec, problem_vec)
        actual = 1 if is_correct else 0
        error = actual - predicted
        student_vec += lr * error * problem_vec
    return student_vec.tolist()

def recommend_questions(student_id, seen_ids=None, top_n=20):
    if seen_ids is None:
        seen_ids = []
    seen_ids_set = {str(x) for x in seen_ids}
    student_vec = None
    if str(student_id) in student_factors:
        student_vec = np.array(student_factors[str(student_id)])
    else:
        try:
            doc_ref = db.collection('students').document(str(student_id)).get()
            if doc_ref.exists:
                doc_data = doc_ref.to_dict()
                if 'factors' in doc_data:
                    student_vec = np.array(doc_data['factors'])
        except Exception as e:
            print(f"Error fetching student factors from Firestore: {e}")

    if student_vec is None:
        return []

    scores = {}
    for problem_id, factors in problem_factors.items():
        if str(problem_id) in seen_ids_set:
            continue
        problem_vec = np.array(factors)
        predicted_score = np.dot(student_vec, problem_vec)
        scores[problem_id] = predicted_score
    sorted_problems = sorted(scores.items(), key=lambda x: x[1])
    return [problem_id for problem_id, score in sorted_problems[:top_n]]


def generate_question(original_question_text):
    prompt = f"""
Here is a math question written by a professional math teacher:

{original_question_text}

Your job is to generate 1 NEW question that:
- Tests exactly the same math skill
- Has different numbers or values
- Is similar in difficulty
- Is written in clean plain English (no LaTeX)

Return ONLY a JSON object in this exact format, nothing else:
{{
  "question": "question text here",
  "A": "option A text",
  "B": "option B text",
  "C": "option C text",
  "D": "option D text",
  "correct": "A"
}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    result = response.choices[0].message.content
    # Clean the result
    result = result.strip()
    result = result.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Extract JSON if wrapped in markdown
    if '```json' in result:
        result = result.split('```json')[1].split('```')[0].strip()
    elif '```' in result:
        result = result.split('```')[1].split('```')[0].strip()
    try:
        return json.loads(result)
    except Exception as e:
        print(f"Groq bad response: {result}")
        return None

def save_attempt(student_id, problem_id, is_correct):
    db.collection('attempts').add({
        'student_id': str(student_id),
        'problem_id': str(problem_id),
        'is_correct': is_correct
    })

# ── Endpoints ──────────────────────────────────────

@app.route('/build_student', methods=['POST'])
def build_student():
    data = request.json
    student_id = data['student_id']
    attempts = data['attempts']
    factors = build_student_factors(attempts)
    db.collection('students').document(student_id).set({
        'student_id': student_id,
        'factors': factors
    })
    return jsonify({'status': 'built', 'student_id': student_id})

@app.route('/attempt', methods=['POST'])
def attempt():
    data = request.json
    student_id = data['student_id']
    problem_id = data['problem_id']
    is_correct = data['is_correct']
    
    save_attempt(student_id, problem_id, is_correct)
    
    # Re-calculate student factors with the new attempt
    try:
        attempts_ref = db.collection('attempts').where('student_id', '==', str(student_id)).stream()
        attempts = []
        for doc in attempts_ref:
            attempts.append(doc.to_dict())
        
        if attempts:
            factors = build_student_factors(attempts)
            db.collection('students').document(str(student_id)).set({
                'student_id': str(student_id),
                'factors': factors
            })
            print(f"Successfully updated SVD factors for student {student_id}")
    except Exception as e:
        print(f"Error updating student factors: {e}")
        
    return jsonify({'status': 'saved'})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    student_id = data['student_id']
    seen_ids = [str(sid) for sid in data.get('seen_ids', [])]
    goal = data.get('goal')
    
    # Map of goals to allowed subjects
    GOAL_SUBJECT_MAPPING = {
        "Class 9": ["BIDMAS", "Place Value", "Time", "Reflection", "Length Units", "Area of Simple Shapes", "Rounding to the Nearest Whole (10, 100, etc)"],
        "Class 10": ["Linear Equations", "Quadratic Equations", "Factorising into a Double Bracket", "Volume and Capacity Units", "Simultaneous Equations", "Inequalities on Number Lines"],
        "JEE": ["Equation of a Circle", "Transformations of functions", "Sketching from Factorised Form", "Solving Quadratic Inequalities", "Perpendicular Lines"],
        "NEET": ["Substitution into Formula", "Real Life Graphs", "Trial and Improvement", "Averages (mean, median, mode) from a List of Data", "Percentages of an Amount", "Multiplying and Dividing with Decimals"]
    }
    
    allowed_subjects = GOAL_SUBJECT_MAPPING.get(goal, [])
    
    # Fetch 500 recommendations so we have a wide pool of subjects to filter from
    recommended_ids = recommend_questions(student_id, seen_ids=seen_ids, top_n=500)
    final_questions = []
    generated = None
    for problem_id in recommended_ids:
        if str(problem_id) in seen_ids:
            continue
        row = question_bank[question_bank['QuestionId'] == int(problem_id)]
        if len(row) == 0:
            continue
            
        subject_name = row['SubjectName'].values[0]
        # Filter by allowed subjects for the student's selected goal
        if allowed_subjects and subject_name not in allowed_subjects:
            continue
            
        original_text = row['QuestionText'].values[0]
        generated = generate_question(original_text)
        if generated is not None:
            generated['problem_id'] = problem_id
            final_questions.append(generated)
            # Break as soon as we have successfully generated 1 matching question for the client
            break
            
    # Fallback: if no question was generated (all SVD recommendations failed or were seen/excluded),
    # select from the question bank directly for the allowed subjects.
    if len(final_questions) == 0 and allowed_subjects:
        print("Fallback: SVD recommendations did not yield any unseen questions. Selecting directly from bank...")
        candidates = question_bank[question_bank['SubjectName'].isin(allowed_subjects)]
        unseen_candidates = candidates[~candidates['QuestionId'].astype(str).isin(seen_ids)]
        
        if len(unseen_candidates) > 0:
            row = unseen_candidates.sample(n=1)
            problem_id = str(row['QuestionId'].values[0])
            original_text = row['QuestionText'].values[0]
            generated = generate_question(original_text)
            if generated is not None:
                generated['problem_id'] = problem_id
                final_questions.append(generated)
                
    return jsonify({'questions': final_questions})
@app.route('/onboard', methods=['POST'])
def onboard():
    data = request.json
    goal = data['goal']
    confidence = data['confidence']
    prompt = f"""
You are a math teacher creating an onboarding quiz.

Student goal: {goal} (Class 9 / Class 10 / JEE / NEET)
Student confidence: {confidence} (Beginner / Average / Strong)

Generate 5 multiple choice questions across 5 different math topics
appropriate for this student's goal and confidence level.

Return ONLY a JSON array in this exact format, nothing else:
[
  {{
    "question": "question text here",
    "A": "option A",
    "B": "option B",
    "C": "option C",
    "D": "option D",
    "correct": "A",
    "topic": "topic name here"
  }}
]
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    result = response.choices[0].message.content
    questions = json.loads(result)
    return jsonify({'questions': questions})

if __name__ == '__main__':
    app.run(debug=True)