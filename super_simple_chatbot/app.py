from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load a pre-trained language model
nlp = pipeline("text-generation", model="facebook/blenderbot-400M-distill")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get('user_input')
    print(f"Received user input: {user_input}")  # Debug print

    try:
        response = nlp(user_input, max_length=50, num_return_sequences=1)
        print(f"Generated response: {response[0]['generated_text']}")  # Debug print
        return jsonify({'response': response[0]['generated_text']})
    except Exception as e:
        print(f"Error: {e}")  # Debug print
        return jsonify({'response': 'Sorry, I encountered an error.'})

if __name__ == '__main__':
    app.run(debug=True)