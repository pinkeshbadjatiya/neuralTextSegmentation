#!/usr/bin/python

from flask import Flask
from flask import render_template
from flask import request, redirect, url_for
import subprocess
import re

app = Flask(__name__)


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == "POST" and request.form and 'document' in request.form:
        data = request.form['document']
        if len(data) > 0:
            return redirect(url_for('.main', document=request.form['document']))
    with open("input.xml") as f:
        lines = f.readlines()
        document = "".join([line.decode("UTF-8") for line in lines])
    return render_template("upload.html", sample_doc=document)


@app.route("/")
def main():
    document = "Sample text!"
    if request.args and 'document' in request.args:
        document = request.args['document']  # counterpart for url_for()
    else:
        with open("input.xml") as f:
            lines = f.readlines()
            document = "".join([line.decode("UTF-8") for line in lines])
    return render_template("document.html", original_document=document, segmented_document="", segments=False)


@app.route("/segments")
def segments():
    TEMP_FILENAME="last_uploaded_document.xml"
    if request.args and 'document' in request.args:
        original_document = request.args['document']  # counterpart for url_for()
        with open(TEMP_FILENAME, 'w+') as f:
            f.write(original_document.encode("UTF-8"))
        subprocess.call(['java', '-cp', 'TopicSegmentation.jar', 'wiki.topicsegmentor.LinearTopicSegmentor', TEMP_FILENAME, 'output.xml'])
        with open("output.xml") as f:
            lines = f.readlines()
            document = "".join([line.decode("UTF-8") for line in lines])

            # Process document tags
            document = re.sub(r'<topic_segment>', "<p class='segment'>", document, flags=re.IGNORECASE)
            document = re.sub(r'</topic_segment>', "</p>", document, flags=re.IGNORECASE)
        return render_template("document.html", segmented_document=document, original_document=original_document, segments=True)
    else:
        return "No document submitted! Please submit a document"



if __name__ == "__main__":
    #app.run(host='0.0.0.0', threaded=True)
    app.run(host='0.0.0.0')
    #from gevent.wsgi import WSGIServer
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()
