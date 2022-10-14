from flask import Flask, render_template, request
import predictm


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def predict():
    jpg_file = request.files['my_image']
    file_path = jpg_file.filename
    jpg_file.save(file_path)


    pred = predictm.classify_image(file_path)

    return render_template("index.html", result=pred['class'], probability=round(pred['class_probability'], 2),
                           img_path=file_path)


if __name__ == "__main__":
    app.run(port=5000, debug=True, host='0.0.0.0')
