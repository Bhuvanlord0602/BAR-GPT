from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    input_text = request.form["msg"]
    response = get_chat_response(input_text)
    return jsonify({"response": response})

def get_chat_response(text):
    global chat_history
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history, new_user_input_ids], dim=-1) if chat_history is not None else new_user_input_ids

    # generate a response while limiting the total chat history to 1000 tokens
    chat_history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # decode and return the response
    response = tokenizer.decode(chat_history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    app.run()
