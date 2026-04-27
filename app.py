from flask import Flask, request, jsonify, render_template

print("Step 1: Starting app.py...")

try:
    from predict import predict_news
    print("Step 2: Imported predict.py successfully")
except Exception as e:
    print("Error importing predict:", e)

app = Flask(__name__)

@app.route('/')
def home():
    print("Home route accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route called")
    text = request.form['news']
    result = predict_news(text)
    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api():
    data = request.get_json()
    result = predict_news(data['news'])
    return jsonify({'prediction': result})

if __name__ == '__main__':
    print("Step 3: Running Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
