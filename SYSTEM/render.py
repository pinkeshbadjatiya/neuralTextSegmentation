from flask import Flask
app = Flask(__name__)

@app.route('/')
def render_document():
    return render_template('document.html', document=document)
