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
    'Limits & Continuity': ['Limits by Substitution', 'L Hopital Rule', 'Continuity at a Point', 'Sandwich Theorem'],
    'Differentiation': ['First Principles', 'Chain Rule', 'Product Rule', 'Quotient Rule', 'Implicit Differentiation'],
    'Integration': ['Indefinite Integrals', 'Definite Integrals', 'Integration by Substitution', 'Integration by Parts'],
    'Differential Equations': ['Variable Separable', 'Homogeneous Differential Equations', 'Linear Differential Equations'],
    'Matrices & Determinants': ['Matrix Operations', 'Determinants', 'Inverse of a Matrix', 'System of Equations'],
    'Vectors & 3D Geometry': ['Dot Product', 'Cross Product', 'Lines in 3D', 'Planes in 3D', 'Angle Between Lines and Planes'],
    'Coordinate Geometry': ['Parabola', 'Ellipse', 'Hyperbola', 'Circle'],
    'Straight Lines & Circles': ['Equation of a Line', 'Distance Formula', 'Equation of a Circle', 'Tangent to a Circle'],
    'Probability & Statistics': ['Bayes Theorem', 'Binomial Distribution', 'Conditional Probability', 'Mean and Variance'],
    'Complex Numbers': ['Argument and Modulus', 'Polar Form', 'De Moivres Theorem', 'Roots of Unity'],
    'Sequences & Series': ['AP and GP', 'Sum of Series', 'Binomial Theorem', 'Mathematical Induction'],
    'Trigonometry': ['Trigonometric Identities', 'Inverse Trigonometry', 'Solutions of Triangles', 'Heights and Distances']
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
    if p_known < 0.45:
        difficulty = "medium"
        difficulty_guide = "multi-step problem requiring careful application of concepts"
    elif p_known < 0.60:
        difficulty = "medium-hard"
        difficulty_guide = "problem requiring strong conceptual understanding and multiple steps"
    elif p_known < 0.75:
        difficulty = "hard"
        difficulty_guide = "challenging problem similar to JEE Mains level"
    elif p_known < 0.85:
        difficulty = "very hard"
        difficulty_guide = "difficult problem similar to JEE Advanced level"
    else:
        difficulty = "expert"
        difficulty_guide = "highly challenging problem at JEE Advanced top percentile level"

    subtopics = skill_map.get(skill, [skill])
    subtopic = random.choice(subtopics)

    json_template = '{"question": "question text here", "option_a": "first option", "option_b": "second option", "option_c": "third option", "option_d": "fourth option", "correct": "A or B or C or D", "explanation": "clear step by step solution"}'

    prompt = f"""Generate a {difficulty} JEE Maths multiple choice question on {skill} — specifically on {subtopic}.
Difficulty guide: {difficulty_guide}

This is for a Class 11-12 Indian student preparing for JEE (Joint Entrance Examination).
Questions must require multi-step thinking — never straightforward single-step calculations.

Strict rules:
- Exactly one correct answer
- All 4 options must be plausible — no obviously wrong options
- No images, diagrams or tables required
- Write ALL mathematical expressions wrapped in [MATH]...[/MATH] tags
- Use proper LaTeX inside the tags
- Examples:
  - Fractions: [MATH]\\frac{{1}}{{6}}[/MATH]
  - Powers: [MATH]x^{{2}}[/MATH]
  - Limits: [MATH]\\lim_{{x \\to 0}}[/MATH]
  - Square roots: [MATH]\\sqrt{{x}}[/MATH]
  - Trigonometry: [MATH]\\sin(x)[/MATH]
- Never use LaTeX outside of [MATH]...[/MATH] tags
- Plain text parts of the question should remain plain text
- Use clean mathematical notation
- Question must be unambiguous

Return ONLY this JSON format, no markdown, no explanation:
{json_template}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content
        content = content.replace('```json', '').replace('```', '').strip()

        # Fix LaTeX backslashes before JSON parsing
        import re
        def fix_json_backslashes(s):
            # Replace all backslashes not followed by valid JSON escape chars
            return re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', s)

        content = fix_json_backslashes(content)
        question = json.loads(content)
        question['skill'] = skill
        question['subtopic'] = subtopic
        return question
    except Exception as e:
        print(f"Groq failed: {e}")
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
    topic = data['topic']

    student = load_student(student_id)
    p_known = student['skills'].get(topic, 0.3)

    question = generate_question_v2(topic, p_known)

    if question is None:
        return jsonify({'error': 'Failed to generate question'}), 500

    return jsonify(question)

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

@app.route('/profile', methods=['GET'])
def profile():
    student_id = request.args.get('student_id')
    student = load_student(student_id)
    return jsonify({
        'student_id': student_id,
        'skills': student['skills']
    })

# ── Run Server ────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)