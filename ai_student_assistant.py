import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from google import genai
import os


# ============================================================
# CODOMAX MODULE 6
# AI STUDENT PERFORMANCE PREDICTOR & STUDY ASSISTANT
# ============================================================

print("=" * 60)
print("     AI STUDENT PERFORMANCE PREDICTOR & STUDY ASSISTANT")
print("=" * 60)


# ============================================================
# 1. STUDENT PERFORMANCE DATASET
# ============================================================

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 1, 3, 6, 7, 4, 5, 8, 2],
    "Attendance": [65, 70, 75, 80, 85, 90, 95, 60, 72, 88, 92, 78, 82, 96, 68],
    "Previous_Marks": [45, 50, 55, 60, 65, 70, 80, 40, 52, 72, 78, 58, 63, 85, 48],
    "Assignment_Score": [50, 55, 60, 65, 70, 75, 85, 45, 58, 78, 82, 62, 68, 90, 52],
    "Performance": [
        "Needs Improvement",
        "Needs Improvement",
        "Average",
        "Average",
        "Good",
        "Good",
        "Excellent",
        "Needs Improvement",
        "Average",
        "Good",
        "Excellent",
        "Average",
        "Good",
        "Excellent",
        "Needs Improvement"
    ]
}

df = pd.DataFrame(data)


# ============================================================
# 2. PREPARE DATA
# ============================================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignment_Score"
    ]
]

y = df["Performance"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# ============================================================
# 3. TRAIN MACHINE LEARNING MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4
)

model.fit(X_train, y_train)


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 5. GEMINI AI CONNECTION
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("\nGemini API key not found.")
    print("Please set the GEMINI_API_KEY environment variable.")
else:
    client = genai.Client(api_key=api_key)


# ============================================================
# 6. AI STUDY ASSISTANT
# ============================================================

def ai_study_assistant(topic):

    prompt = f"""
    You are an AI Study Assistant for a second-year engineering student.

    Study Topic: {topic}

    Provide:

    1. Simple Explanation
    2. 5 Key Points
    3. Short Summary
    4. 3 Practice Questions
    5. One Study Tip

    Use simple and clear English.
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# ============================================================
# 7. FINAL AI STUDENT ASSISTANT
# ============================================================

def final_ai_student_assistant():

    print("\nEnter Student Details:")

    study_hours = float(input("Study Hours: "))
    attendance = float(input("Attendance (%): "))
    previous_marks = float(input("Previous Marks: "))
    assignment_score = float(input("Assignment Score: "))

    student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignment_Score": [assignment_score]
    })

    prediction = model.predict(student)

    performance = label_encoder.inverse_transform(prediction)[0]

    print("\nPredicted Performance:", performance)

    topic = input("\nEnter a study topic for AI assistance: ")

    if not api_key:
        print("\nGemini API key is not configured.")
        return

    print("\nGenerating AI study material...\n")

    study_material = ai_study_assistant(topic)

    print(study_material)


# ============================================================
# 8. RUN PROJECT
# ============================================================

final_ai_student_assistant()
