from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Sample gestures and their meanings
gestures = {
    "Seven": "Approval",
    "Thumbs Up": "Peace",
    "Peace Sign": "Seven",
    "Wave": "Power",
    "Clenched Fist": "Hello",
    

}

@app.route("/")
def index():
    return render_template("index.html", gestures=gestures)

@app.route("/check", methods=["POST"])
def check():
    selected_gesture = request.form.get("gesture")
    selected_meaning = request.form.get("meaning")
    correct_meaning = gestures.get(selected_gesture)

    if correct_meaning == selected_meaning:
        return jsonify({"result": "correct", "message": "Correct Match!"})
    else:
        return jsonify({"result": "wrong", "message": "Wrong Match!"})

if __name__ == "__main__":
    app.run(debug=True)
