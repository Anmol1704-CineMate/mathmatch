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
    if firebase_key:
        try:
            cred_dict = json.loads(firebase_key)
            if 'private_key' in cred_dict:
                cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            print(f"Error loading Firebase credentials from env: {e}")

db = firestore.client() if firebase_admin._apps else None

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
P_GUESS = 0.25
P_SLIP = 0.1
P_LEARN = 0.1

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

def generate_question_v2(skill, p_known):
    
    if p_known < 0.35:
        difficulty = "very easy"
        difficulty_guide = "single step problem, direct formula application, no multi-step thinking required"
    elif p_known < 0.45:
        difficulty = "easy"
        difficulty_guide = "straightforward calculation involving one concept, minimal steps"
    elif p_known < 0.52:
        difficulty = "easy-medium"
        difficulty_guide = "two step problem using a familiar concept, slightly more thinking required"
    elif p_known < 0.60:
        difficulty = "medium"
        difficulty_guide = "requires applying the concept in a slightly new context, 2-3 steps"
    elif p_known < 0.68:
        difficulty = "medium-hard"
        difficulty_guide = "multi step problem requiring some deeper thinking, not immediately obvious"
    elif p_known < 0.75:
        difficulty = "hard"
        difficulty_guide = "combines two concepts together, exam style question, requires planning"
    elif p_known < 0.85:
        difficulty = "very hard"
        difficulty_guide = "complex multi step problem, competitive exam level, requires strong understanding"
    else:
        difficulty = "expert"
        difficulty_guide = "mastery level question, hardest possible, olympiad or top exam standard"

    prompt = f"""Generate a {difficulty} {skill} multiple choice question for a Class 10 student aged 14-16.
Difficulty guide: {difficulty_guide}

Strict rules:
- Exactly one correct answer
- All 4 options must be plausible — no obviously wrong options
- Solvable without a calculator
- No images, diagrams or tables required
- Use clean simple mathematical notation
- Question must be unambiguous

Return ONLY a JSON object, no markdown, no explanation:
{{
    "question": "question text here",
    "option_a": "first option",
    "option_b": "second option",
    "option_c": "third option",
    "option_d": "fourth option",
    "correct": "A or B or C or D",
    "explanation": "clear step by step solution showing why the answer is correct"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        content = response.choices[0].message.content
        content = content.replace('```json', '').replace('```', '').strip()
        question = json.loads(content)
        return question
    except:
        return None

# ── Firebase Functions ────────────────────────────────────────
def save_student(student):
    if db:
        db.collection('students').document(student['student_id']).set(student)

def load_student(student_id):
    if db:
        doc = db.collection('students').document(student_id).get()
        if doc.exists:
            return doc.to_dict()
    return create_student_profile(student_id)

# ── Flask Endpoints ───────────────────────────────────────────
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    student_id = data['student_id']
    seen_ids = data.get('seen_ids', [])

    student = load_student(student_id)
    skill = pick_skill(student)
    p_known = student['skills'][skill]

    # Step 1 — Try Groq first (PRIMARY)
    groq_question = generate_question_v2(skill, p_known)

    if groq_question is not None:
        return jsonify({
            'source': 'groq',
            'skill': skill,
            'question': groq_question['question'],
            'option_a': groq_question['option_a'],
            'option_b': groq_question['option_b'],
            'option_c': groq_question['option_c'],
            'option_d': groq_question['option_d'],
            'correct': groq_question['correct'],
            'explanation': groq_question['explanation'],
            'question_id': None
        })

    # Step 2 — Groq failed, try Eedi (BACKUP)
    eedi_question = pick_question(skill, seen_ids)

    if eedi_question is not None:
        return jsonify({
            'source': 'eedi',
            'skill': skill,
            'question': eedi_question['QuestionText'],
            'option_a': eedi_question['AnswerAText'],
            'option_b': eedi_question['AnswerBText'],
            'option_c': eedi_question['AnswerCText'],
            'option_d': eedi_question['AnswerDText'],
            'correct': eedi_question['CorrectAnswer'],
            'explanation': None,
            'question_id': int(eedi_question['QuestionId'])
        })

    # Step 3 — Both failed
    return jsonify({'error': 'No question available'}), 500

print("✅ /recommend endpoint ready!")

@app.route('/attempt', methods=['POST'])
def attempt():
    data = request.json
    student_id = data['student_id']
    skill = data.get('skill', 'Algebra')
    is_correct = data['is_correct']

    student = load_student(student_id)
    old_score = student['skills'].get(skill, 0.3)
    new_score = update_skill(old_score, is_correct)
    student['skills'][skill] = new_score
    save_student(student)

    return jsonify({
        'skill': skill,
        'old_score': old_score,
        'new_score': new_score,
        'mastered': new_score >= 0.85
    })

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'GET':
        student_id = request.args.get('student_id')
    else:
        data = request.json or {}
        student_id = data.get('student_id')
    student = load_student(student_id)
    return jsonify({
        'student_id': student_id,
        'skills': student['skills']
    })

# ── Run Server ────────────────────────────────────────────────
if __name__ == '__main__':
    # Test all 8 levels
    test_scores = [0.30, 0.38, 0.48, 0.55, 0.63, 0.70, 0.78, 0.87]

    for score in test_scores:
        try:
            result = generate_question_v2('Algebra', p_known=score)
            if result:
                print(f"P(known)={score} → {result['question'][:60]}...")
            else:
                print(f"P(known)={score} → ⚠️ Groq returned bad JSON — retrying skipped")
        except Exception as e:
            print(f"P(known)={score} → ⚠️ Groq call failed: {e}")
        print()

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)