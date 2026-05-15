from flask import Flask, request, jsonify, render_template
import pickle
from main import clean_text

app = Flask(__name__)

# Model load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    
    clean = clean_text(review)
    vectorized = vectorizer.transform([clean])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    if prediction == 1:
        result = {
            'sentiment': 'POSITIVE',
            'confidence': f"{probability[1]*100:.1f}%"
        }
    else:
        result = {
            'sentiment': 'NEGATIVE', 
            'confidence': f"{probability[0]*100:.1f}%"
        }
    
    return jsonify(result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    