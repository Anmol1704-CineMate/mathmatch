import json
import numpy as np
import pandas as pd

with open('problem_factors.json', 'r') as f:
    problem_factors = json.load(f)

with open('student_factors.json', 'r') as f:
    student_factors = json.load(f)

question_bank = pd.read_csv('question_bank.csv')

# Let's inspect some student vectors
student_keys = list(student_factors.keys())
print("Number of students:", len(student_keys))
print("Sample student vector (first 5 elements of student #184):", student_factors[student_keys[0]][:5])

# Let's compute average dot product scores for a student
student_vec = np.array(student_factors[student_keys[0]])
scores = []
for pid, factors in problem_factors.items():
    p_vec = np.array(factors)
    score = np.dot(student_vec, p_vec)
    scores.append((pid, score))

scores.sort(key=lambda x: x[1])

print("\nTop 5 lowest scoring questions (should be hardest if ascending, or easiest?):")
for pid, score in scores[:5]:
    row = question_bank[question_bank['QuestionId'] == int(pid)]
    if len(row) > 0:
        print(f"ID: {pid}, Score: {score:.4f}, Subject: {row['SubjectName'].values[0]}, Text: {row['QuestionText'].values[0][:100]}")

print("\nTop 5 highest scoring questions (should be easiest):")
for pid, score in scores[-5:]:
    row = question_bank[question_bank['QuestionId'] == int(pid)]
    if len(row) > 0:
        print(f"ID: {pid}, Score: {score:.4f}, Subject: {row['SubjectName'].values[0]}, Text: {row['QuestionText'].values[0][:100]}")
