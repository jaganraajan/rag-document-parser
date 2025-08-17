from flask import Flask, render_template

app = Flask(__name__, template_folder="flask-template", static_folder="flask-template")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")
