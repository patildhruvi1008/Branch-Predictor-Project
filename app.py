from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/branch_predictor.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data
    input_data = [float(x) for x in request.form.values()]
    input_df = pd.DataFrame([input_data], columns=['Category', 'JEE Marks', '10th Marks', '12th Marks'])
    
    # Predict branch
    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction=f"Predicted Branch: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
