from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///face.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Embeddings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    second_name = db.Column(db.String(100), nullable=False)
    # person_embedding = db.Column(db.Text, nullable=False)
    # data_add = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Embeddings %r>' % self.id


@app.route('/add_person', methods=['POST', 'GET'])
def add_person():
    if request.method == "POST":
        first_name = request.form['first_name']
        second_name = request.form['second_name']

        embedding = Embeddings(first_name=first_name, second_name=second_name)

        try:
            db.session.add(embedding)
            db.session.commit()
            return redirect("/home")
        except:
            return "При добавлении человека произошла ошибка"
    else:
        return render_template("add_person.html")


@app.route('/')
@app.route('/home')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/saved_persons')
def saved_persons():
    return render_template("saved_persons.html")


@app.route('/user/<string:name>/<int:id>')
def user(name, id):
    return "User page " + name + " - " + str(id)


if __name__ == "__main__":
    app.run(debug=True)
