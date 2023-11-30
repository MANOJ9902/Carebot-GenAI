# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import nltk
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

with open(r"C:\Users\MANOJ KUMAR R\Downloads\Health-Care-Chatbot-main (1)\Health-Care-Chatbot-main\intents.json") as json_file:
    intents = json.load(json_file)

words = pickle.load(open(r"C:\Users\MANOJ KUMAR R\Downloads\Health-Care-Chatbot-main (1)\Health-Care-Chatbot-main\words.pkl", "rb"))
classes = pickle.load(open(r"C:\Users\MANOJ KUMAR R\Downloads\Health-Care-Chatbot-main (1)\Health-Care-Chatbot-main\classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]
        for i in list_of_intents:
            if i["tag"] == tag:
                result = random.choice(i["responses"])
                break
        return result
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase or ask another question?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    user_message = request.form["user_message"]
    ints = predict_class(user_message)
    response = get_response(ints, intents)
    return jsonify({"bot_response": response})

if __name__ == "__main__":
    app.run(debug=True)
