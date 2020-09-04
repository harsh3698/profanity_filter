from flask import Flask, render_template, url_for, redirect, request, flash, send_file
import os
from secrets import token_hex
import pandas as pd
from model.predict import predict, initialise, predict_text


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == "csv"


app = Flask(__name__)
upfolder = './model'
app.config['UPLOAD_FOLDER'] = upfolder

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", token_hex(16))


@app.route('/', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            text = request.form['text']
            if text:
                val = predict_text(text, model, offensive_words,
                                   profane_model, profane_vectorizer)
                if val:
                    flash(val, 'info')
                else:
                    flash(val, 'success')
                return redirect(request.url)
        except:
            file = request.files['file']
            if file:
                if allowed_file(file.filename):
                    val = predict(file, model, offensive_words,
                                  upfolder, profane_model, profane_vectorizer)
                    if val:
                        flash(val, 'info')
                        return redirect(request.url)
                    else:
                        flash(
                            'File successfully uploaded. Kindly wait for results. It could take a while', 'success')
                        return redirect(url_for('results'))
                else:
                    flash('Only CSV files are allowed', 'info')
                    return redirect(request.url)
                return redirect(request.url)
    else:
        return render_template('upload.html')


@app.route('/results')
def results():
    return send_file(os.path.join(upfolder, 'result.csv'),
                     mimetype='text/csv',
                     attachment_filename='result.csv',
                     as_attachment=True)


@app.errorhandler(Exception)
def handle_error(e):
    return render_template("errorpage.html")


if __name__ == '__main__':
    model, offensive_words, profane_model, profane_vectorizer = initialise(
        upfolder)

    # ensure host is 0.0.0.0 when going for docker and aws
    # for AWS ensure to use 80:5000 to map port 5000 to port 80 of aws
    app.run(port=5000, debug=False, host='0.0.0.0')
    # app.run(port=5000, debug=True)
