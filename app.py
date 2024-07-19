from flask import Flask, request, jsonify
from flask_cors import CORS
from Model.PersonaChatbot.PersonaChatbot import PersonaChatbot

app = Flask(__name__)
CORS(app)

persona_chatbots = {}  # Dictionary to store PersonaChatbot instances for each session

@app.route('/api/chat', methods=['POST'])
def fitness_chat():
    try:
        request_json = request.get_json()

        session_id = request_json.get('session_id')
        user_input = request_json.get('message')

        if not session_id or not user_input:
            return jsonify({"error": "session_id and message parameters are required."}), 400

        # Create or retrieve PersonaChatbot instance for the session
        if session_id not in persona_chatbots:
            persona_chatbots[session_id] = PersonaChatbot(f"Healty Diet Assistant")

        persona_chatbot = persona_chatbots[session_id]

        # Set the user input for the chat session
        persona_chatbot.user_input = user_input

        # Choose the next agent for the session
        _, agent_name = persona_chatbot.choose_next_agent()

        # Get response from the chosen agent
        answer = persona_chatbot.run_agent_by_name(agent_name=agent_name)

        response = jsonify({"response": answer})
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response, 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0")