import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier

# Load dataset from Vishal's GitHub raw link
url = "https://raw.githubusercontent.com/Garvitpujari/ml-model-deploy/main/diabetes.csv"
df = pd.read_csv(url)

# Split into features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Prediction function
def predict_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                     BMI, DiabetesPedigreeFunction, Age):
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                BMI, DiabetesPedigreeFunction, Age]],
                              columns=X.columns)
    prediction = model.predict(input_data)[0]
    return "Diabetic" if prediction == 1 else "Not Diabetic"

# Define Gradio inputs
inputs = [
    gr.Number(label="Pregnancies"),
    gr.Number(label="Glucose"),
    gr.Number(label="BloodPressure"),
    gr.Number(label="SkinThickness"),
    gr.Number(label="Insulin"),
    gr.Number(label="BMI"),
    gr.Number(label="DiabetesPedigreeFunction"),
    gr.Number(label="Age"),
]

# Output
output = gr.Text(label="Prediction")

# Create Gradio Interface
app = gr.Interface(fn=predict_diabetes, inputs=inputs, outputs=output, title="Diabetes Prediction App")

# Run the app
if _name_ == "_main_":
    app.launch()
