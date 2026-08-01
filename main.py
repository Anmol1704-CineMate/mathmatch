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

# ── Razorpay Setup ───────────────────────────────────────────
import razorpay

RAZORPAY_KEY_ID = "rzp_live_TK9JxtlFsAsp1k"
RAZORPAY_KEY_SECRET = "DodhRxkkIPoAEeE30OBqYq5H"

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

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

# ── Student Functions ──────────────────────────────────────────
def create_student_profile(student_id):
    return {
        'student_id': student_id,
        'skills': {
            'Limits & Continuity': 60,
            'Differentiation': 60,
            'Integration': 60,
            'Differential Equations': 60,
            'Matrices & Determinants': 60,
            'Vectors & 3D Geometry': 60,
            'Coordinate Geometry': 60,
            'Straight Lines & Circles': 60,
            'Probability & Statistics': 60,
            'Complex Numbers': 60,
            'Sequences & Series': 60,
            'Trigonometry': 60
        },
        'streaks': {
            'Limits & Continuity': 0,
            'Differentiation': 0,
            'Integration': 0,
            'Differential Equations': 0,
            'Matrices & Determinants': 0,
            'Vectors & 3D Geometry': 0,
            'Coordinate Geometry': 0,
            'Straight Lines & Circles': 0,
            'Probability & Statistics': 0,
            'Complex Numbers': 0,
            'Sequences & Series': 0,
            'Trigonometry': 0
        },
        'is_pro': False
    }

def update_streak(mastery, streak, is_correct):
    if is_correct:
        if streak >= 0:
            streak += 1
        else:
            streak = 1
    else:
        if streak <= 0:
            streak -= 1
        else:
            streak = -1
    if streak == 3:
        mastery = min(95, mastery + 5)
        streak = 0
    elif streak == -3:
        mastery = max(10, mastery - 5)
        streak = 0
    return mastery, streak

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

def generate_question_v2(skill, mastery, seen_questions=[]):
    if mastery < 40:
        difficulty = "medium"
        difficulty_guide = "multi-step problem requiring careful application of concepts"
    elif mastery < 55:
        difficulty = "medium-hard"
        difficulty_guide = "problem requiring strong conceptual understanding and multiple steps"
    elif mastery < 70:
        difficulty = "hard"
        difficulty_guide = "challenging problem similar to JEE Mains level"
    elif mastery < 85:
        difficulty = "very hard"
        difficulty_guide = "difficult problem similar to JEE Advanced level"
    else:
        difficulty = "expert"
        difficulty_guide = "highly challenging problem at JEE Advanced top percentile level"

    subtopics = skill_map.get(skill, [skill])
    subtopic = random.choice(subtopics)

    json_template = '{"question": "question text here", "option_a": "first option", "option_b": "second option", "option_c": "third option", "option_d": "fourth option", "correct": "A or B or C or D", "explanation": "Step 1: [first step]. Step 2: [second step]. Step 3: [final step and answer]."}'

    seen_text = "; ".join(seen_questions[-5:]) if seen_questions else "none"

    prompt = f"""Generate a {difficulty} JEE Advanced style MCQ on {skill} — specifically {subtopic}.

This must feel like an actual JEE Advanced question:
- Every question must have a DIFFERENT structure and setup from typical questions
- Rotate between these formats: function analysis, inequality problems, area under curve, definite integral properties, limit of sum, continuity & differentiability combined, inverse functions
- Never use the same question template twice in a row
- Vary the mathematical objects used: sometimes use piecewise functions, sometimes parametric, sometimes implicit
- Never ask direct formula application (e.g. "find dy/dx of x²")
- Always combine 2 or more concepts in one question
- Use tricky setups — unusual domains, boundary conditions, composite functions
- Options must be close to each other — a student who almost understands will pick wrong
- The correct answer must require genuine insight, not just calculation

Difficulty guide: {difficulty_guide}

STRICT RULES:
- Exactly one correct answer
- All 4 options must be plausible — no obviously wrong options
- No images, diagrams or tables required
- Write ALL mathematical expressions wrapped in [MATH]...[/MATH] tags using proper LaTeX
- The question MUST be strictly about {skill} only
- Do NOT repeat or closely resemble any of these recently seen questions: {seen_text}

Explanation rules (Kota teacher style):
- Exactly 3 steps
- Each step: show action AND result with actual math
- Max 30 words per step
- Use [MATH]...[/MATH] for all expressions

IMPORTANT: Do NOT repeat or closely resemble any of these recently seen questions: {seen_text}
Generate a completely different question with different numbers, setup, and concept angle.

Return ONLY this JSON, no markdown:
{json_template}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
        content = response.choices[0].message.content
        content = content.replace('```json', '').replace('```', '').strip()

        # Fix LaTeX backslashes before JSON parsing
        import re
        def fix_json_backslashes(s):
            # Double any single backslash that is not followed by " and not preceded by \
            return re.sub(r'(?<!\\)\\(?!")', r'\\\\', s)

        content = fix_json_backslashes(content)
        question = json.loads(content)
        question['skill'] = skill
        question['subtopic'] = subtopic
        return question
    except Exception as e:
        print(f"Groq failed: {e}")
        print(f"Raw content was: {content if 'content' in dir() else 'no content'}")
        return None

# ── Firebase Functions ────────────────────────────────────────
def save_student(student_id, student=None):
    if student is None:
        student = student_id
        student_id = student.get('student_id', 'guest')
    if db:
        db.collection('students').document(student_id).set(student)

def load_student(student_id):
    if db:
        doc = db.collection('students').document(student_id).get()
        if doc.exists:
            student_data = doc.to_dict()
            if 'skills' not in student_data or not isinstance(student_data['skills'], dict):
                student_data['skills'] = {}
            for skill in skill_map.keys():
                if skill not in student_data['skills']:
                    student_data['skills'][skill] = 30
            if 'streaks' not in student_data or not isinstance(student_data['streaks'], dict):
                student_data['streaks'] = {}
            for skill in skill_map.keys():
                if skill not in student_data['streaks']:
                    student_data['streaks'][skill] = 0
            return student_data
    return create_student_profile(student_id)

# ── Flask Endpoints ───────────────────────────────────────────
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json or {}
        student_id = data.get('student_id') or 'guest'
        topic = data.get('topic') or 'Limits & Continuity'
        seen_questions = data.get('seen_questions', [])

        student = load_student(student_id)
        mastery = 30
        if isinstance(student, dict) and 'skills' in student:
            mastery = student['skills'].get(topic, 30)

        subtopic = random.choice(skill_map.get(topic, [topic]))
        question = generate_question_v2(topic, mastery, seen_questions)

        if question is None:
            question = {
                'question': 'Evaluate the limit: [MATH]\\lim_{x \\to 0} \\frac{\\sin(x)}{x}[/MATH]',
                'option_a': '[MATH]0[/MATH]',
                'option_b': '[MATH]1[/MATH]',
                'option_c': '[MATH]\\infty[/MATH]',
                'option_d': '[MATH]\\text{Undefined}[/MATH]',
                'correct': 'B',
                'explanation': 'Standard trigonometric limit: [MATH]\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = 1[/MATH].',
                'skill': topic,
                'subtopic': subtopic
            }

        return jsonify(question)
    except Exception as e:
        print(f"Error in /recommend: {e}")
        subtopic = 'Standard Limits'
        return jsonify({
            'question': 'Evaluate the limit: [MATH]\\lim_{x \\to 0} \\frac{\\sin(x)}{x}[/MATH]',
            'option_a': '[MATH]0[/MATH]',
            'option_b': '[MATH]1[/MATH]',
            'option_c': '[MATH]\\infty[/MATH]',
            'option_d': '[MATH]\\text{Undefined}[/MATH]',
            'correct': 'B',
            'explanation': 'Standard trigonometric limit: [MATH]\\lim_{x \\to 0} \\frac{\\sin(x)}{x} = 1[/MATH].',
            'skill': 'Limits & Continuity',
            'subtopic': subtopic
        })

@app.route('/attempt', methods=['POST'])
def attempt():
    try:
        data = request.json or {}
        student_id = data.get('student_id') or 'guest'
        skill = data.get('skill') or 'Limits & Continuity'
        is_correct = data.get('is_correct', False)

        student = load_student(student_id)
        mastery = student['skills'][skill]
        streak = student.get('streaks', {}).get(skill, 0)
        new_mastery, new_streak = update_streak(mastery, streak, is_correct)
        student['skills'][skill] = new_mastery
        if 'streaks' not in student:
            student['streaks'] = {}
        student['streaks'][skill] = new_streak
        save_student(student_id, student)
        return jsonify({'new_mastery': new_mastery, 'streak': new_streak, 'mastered': new_mastery >= 85})
    except Exception as e:
        print(f"Error in /attempt: {e}")
        return jsonify({'new_mastery': 30, 'streak': 0, 'mastered': False})

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    try:
        if request.method == 'GET':
            student_id = request.args.get('student_id') or 'guest'
        else:
            data = request.json or {}
            student_id = data.get('student_id') or 'guest'
        student = load_student(student_id)
        return jsonify({
            'student_id': student_id,
            'skills': student.get('skills', {}),
            'is_pro': student.get('is_pro', False)
        })
    except Exception as e:
        print(f"Error in /profile: {e}")
        return jsonify({'student_id': 'guest', 'skills': {}, 'is_pro': False})

@app.route('/create-order', methods=['POST'])
def create_order():
    try:
        order = razorpay_client.order.create({
            'amount': 19900,
            'currency': 'INR',
            'payment_capture': 1
        })
        return jsonify({'order_id': order['id'], 'amount': 19900})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify-payment', methods=['POST'])
def verify_payment():
    try:
        data = request.get_json()
        payment_id = data['razorpay_payment_id']
        order_id = data['razorpay_order_id']
        signature = data['razorpay_signature']
        student_id = data['student_id']

        razorpay_client.utility.verify_payment_signature({
            'razorpay_order_id': order_id,
            'razorpay_payment_id': payment_id,
            'razorpay_signature': signature
        })

        student = load_student(student_id)
        student['is_pro'] = True
        save_student(student_id, student)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ── Run Server ────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)