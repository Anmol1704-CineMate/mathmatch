import os
import json
import random
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import pandas as pd

# ── Flask Setup ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Groq Setup ───────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── Firebase Setup ───────────────────────────────────────────
if not firebase_admin._apps:
    firebase_key = os.environ.get("FIREBASE_CREDENTIALS")
    cred_dict = json.loads(firebase_key)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ── Load Question Bank ───────────────────────────────────────
question_bank = pd.read_csv('question_bank.csv')

# ── Skill Map ────────────────────────────────────────────────
skill_map = {
    'Algebra': ['Linear Equations', 'Writing Expressions',
                'Simplifying Expressions by Collecting Like Terms',
                'Expanding Single Brackets', 'Expanding Double Brackets'],
    'Quadratics': ['Quadratic Equations', 'Completing the Square',
                   'Factorising into a Double Bracket'],
    'Geometry': ['Angles in Triangles', 'Angles in Polygons',
                 'Properties of Triangles', 'Basic Angle Facts (straight line, opposite, around a point, etc)'],
    'Trigonometry': ['Right-angled Triangles (SOHCAHTOA)'],
    'Statistics': ['Averages (mean, median, mode) from a List of Data',
                   'Range and Interquartile Range from a List of Data',
                   'Averages and Range from Frequency Table'],
    'Probability': ['Probability of Single Events',
                    'Combined Events',
                    'Tree Diagrams with Dependent Events'],
    'Fractions': ['Adding and Subtracting Fractions', 'Multiplying Fractions',
                  'Dividing Fractions', 'Simplifying Fractions'],
    'Ratio & Proportion': ['Sharing in a Ratio', 'Direct Proportion',
                           'Indirect (Inverse) Proportion'],
    'Graphs & Coordinates': ['Finding the Equation of a Line',
                             'Gradient as change in y over change in x',
                             'Plotting Lines from Tables of Values'],
    'Sequences': ['Linear Sequences (nth term)', 'Quadratic Sequences'],
    'Indices & Surds': ['Laws of Indices', 'Simplifying Surds',
                        'Operations with Surds'],
    'Mensuration': ['Area of Simple Shapes', 'Volume of Prisms', 'Perimeter',
                    'Surface Area of Prisms']
}

# ── Tag Questions with Skills ─────────────────────────────────
def get_skill(subject_name):
    for skill, subjects in skill_map.items():
        if subject_name in subjects:
            return skill
    return None

question_bank['skill'] = question_bank['SubjectName'].apply(get_skill)

# ── BKT Parameters ───────────────────────────────────────────
P_GUESS = 0.2
P_SLIP = 0.1
P_LEARN = 0.3

# ── BKT Functions ────────────────────────────────────────────
def create_student_profile(student_id):
    profile = {'student_id': student_id, 'skills': {}}
    for skill in skill_map.keys():
        profile['skills'][skill] = 0.3
    return profile

def update_skill(p_known, correct):
    if correct:
        numerator = p_known * (1 - P_SLIP)
        denominator = (p_known * (1 - P_SLIP)) + ((1 - p_known) * P_GUESS)
    else:
        numerator = p_known * P_SLIP
        denominator = (p_known * P_SLIP) + ((1 - p_known) * (1 - P_GUESS))
    p_updated = numerator / denominator
    p_final = p_updated + ((1 - p_updated) * P_LEARN)
    return round(p_final, 4)

def pick_skill(student):
    skills = student['skills']
    weakest_skill = min(skills, key=lambda skill: skills[skill])
    return weakest_skill

def pick_question(skill, seen_ids=[]):
    skill_questions = question_bank[question_bank['skill'] == skill]
    unseen = skill_questions[~skill_questions['QuestionId'].isin(seen_ids)]
    if len(unseen) == 0:
        return None
    question = unseen.sample(1).iloc[0]
    return question

def generate_question(skill, p_known):
    if p_known < 0.4:
        difficulty = "easy"
    elif p_known < 0.7:
        difficulty = "medium"
    else:
        difficulty = "hard"

    prompt = f"""Generate a {difficulty} difficulty {skill} question for a Class 10 / JEE student.

Return ONLY a JSON object in this exact format:
{{
    "question": "question text here",
    "option_a": "option A text",
    "option_b": "option B text",
    "option_c": "option C text",
    "option_d": "option D text",
    "correct": "A or B or C or D",
    "explanation": "why this answer is correct"
}}

No extra text. No markdown. Just the JSON."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        question = json.loads(response.choices[0].message.content)
        return question
    except:
        return None

# ── Firebase Functions ────────────────────────────────────────
def save_student(student):
    db.collection('students').document(student['student_id']).set(student)

def load_student(student_id):
    doc = db.collection('students').document(student_id).get()
    if doc.exists:
        return doc.to_dict()
    else:
        return create_student_profile(student_id)

# ── Flask Endpoints ───────────────────────────────────────────
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    student_id = data['student_id']
    seen_ids = data.get('seen_ids', [])

    student = load_student(student_id)
    skill = pick_skill(student)
    question = pick_question(skill, seen_ids)

    if question is None:
        groq_question = generate_question(skill, student['skills'][skill])
        if groq_question is None:
            return jsonify({'error': 'Could not generate question'}), 500
        return jsonify({
            'source': 'groq',
            'skill': skill,
            'question': groq_question['question'],
            'option_a': groq_question['option_a'],
            'option_b': groq_question['option_b'],
            'option_c': groq_question['option_c'],
            'option_d': groq_question['option_d'],
            'correct': groq_question['correct'],
            'question_id': None
        })

    return jsonify({
        'source': 'eedi',
        'skill': skill,
        'question': question['QuestionText'],
        'option_a': question['AnswerAText'],
        'option_b': question['AnswerBText'],
        'option_c': question['AnswerCText'],
        'option_d': question['AnswerDText'],
        'correct': question['CorrectAnswer'],
        'question_id': int(question['QuestionId'])
    })

@app.route('/attempt', methods=['POST'])
def attempt():
    data = request.json
    student_id = data['student_id']
    skill = data['skill']
    is_correct = data['is_correct']

    student = load_student(student_id)
    old_score = student['skills'][skill]
    new_score = update_skill(old_score, is_correct)
    student['skills'][skill] = new_score
    save_student(student)

    return jsonify({
        'skill': skill,
        'old_score': old_score,
        'new_score': new_score,
        'mastered': new_score >= 0.85
    })

@app.route('/profile', methods=['POST'])
def profile():
    data = request.json
    student_id = data['student_id']
    student = load_student(student_id)
    return jsonify({
        'student_id': student_id,
        'skills': student['skills']
    })

# ── Run Server ────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)