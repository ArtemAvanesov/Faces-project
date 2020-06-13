# from flask import Flask, render_template, url_for, request, redirect
# from flask_sqlalchemy import SQLAlchemy
# import forms
# from datetime import datetime
# import base64
#
# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///face.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)
#
#
# class Embeddings(db.Model):
#     __tablename__ = 'embeddings'
#     __table_args__ = {'extend_existing': True}
#     # LOC_CODE = db.Column(db.Integer, primary_key=True)
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     first_name = db.Column(db.String(100), nullable=False)
#     second_name = db.Column(db.String(100), nullable=False)
#
#     # person_embedding = db.Column(db.Text, nullable=False)
#     # data_add = db.Column(db.DateTime, default=datetime.utcnow)
#
#     def __repr__(self):
#         return '<Embeddings %r>' % self.id
#
#
# @app.route('/')
# @app.route('/home', methods=['POST', 'GET'])
# def index():
#     return render_template("index.html")
#
#
# @app.route('/about')
# def about():
#     if request.method == "POST":
#         b64_string = request.form['cam_photo']
#         image_data = bytes(b64_string, encoding="ascii")
#         # TODO добавить идентификатор для файла
#         with open("temp_photos/imageToSave.jpg", "wb") as fh:
#             fh.write(base64.decodebytes(image_data))
#     return render_template("about.html")
#
#
# @app.route('/person/add', methods=['POST', 'GET'])
# def add_person():
#     form = forms.AddPersonForm(request.form, meta={'csrf': False})
#     if request.method == "POST" and form.validate():
#         first_name = request.form['first_name']
#         second_name = request.form['second_name']
#         b64_string = request.form['photo_hidden']
#         image_data = bytes(b64_string, encoding="ascii")
#         # TODO добавить идентификатор для файла
#         with open("temp_photos/imageToSave.jpg", "wb") as fh:
#             fh.write(base64.decodebytes(image_data))
#         embedding = Embeddings(first_name=first_name, second_name=second_name)
#         try:
#             db.session.add(embedding)
#             db.session.commit()
#             return redirect("/person/saved")
#         except:
#             return "При добавлении человека произошла ошибка"
#     else:
#         return render_template("add_person.html", form=form)
#
#
# @app.route('/person/saved')
# def saved_persons():
#     deleted = request.args.get('deleted', default=False, type=bool)
#     persons = Embeddings.query.order_by(Embeddings.second_name).all()
#     return render_template("saved_persons.html", persons=persons, deleted=deleted)
#
#
# @app.route('/person/delete/<int:id>')
# def delete_person(id):
#     person = Embeddings.query.get_or_404(id)
#     try:
#         db.session.delete(person)
#         db.session.commit()
#         return redirect("/person/saved?deleted=True")
#     except:
#         return "При удалении произошла ошибка!"
#
#
# @app.route('/person/edit/<int:id>', methods=['POST', 'GET'])
# def edit_person(id):
#     person = Embeddings.query.get_or_404(id)
#     if request.method == "POST":
#         person.first_name = request.form['first_name']
#         person.second_name = request.form['second_name']
#         try:
#             db.session.commit()
#             return render_template("edit_person.html", person=person, edited=True)
#         except:
#             return "При редактировании данных произошла ошибка"
#     else:
#         return render_template("edit_person.html", person=person, edited=False)
#
# # Можно убрать, было для тестов
# @app.route('/cam', methods=['POST', 'GET'])
# def cam_func():
#     if request.method == "POST":
#         b64_string = request.form['cam_photo']
#         image_data = bytes(b64_string, encoding="ascii")
#         # TODO добавить идентификатор для файла
#         with open("temp_photos/imageToSave.jpg", "wb") as fh:
#             fh.write(base64.decodebytes(image_data))
#     return render_template("webcam_exp.html")


# @app.route('/user/<string:name>/<int:id>')
# def user(name, id):
#     return "User page " + name + " - " + str(id)


# if __name__ == "__main__":
#     app.run(debug=True)

#-------------------------------------

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from neural_networks.do_embedding import do_embedding
from neural_networks.predict_age import predict_age
from neural_networks.predict_gender import predict_gender
from neural_networks.predict_person import predict_person
from neural_networks.write_annoy import write_annoy
from neural_networks.read_annoy import read_annoy
from work_with_cam.crop_rectangle import crop_rectangle
from keras.models import load_model

embedding_model = load_model('./neural_networks/models/BASE_MODEL.h5')
age_model = load_model('./neural_networks/models/AGE_PART.h5')
gender_model = load_model('./neural_networks/models/FEMALE_MALE_PART.h5')
person_model = load_model('./neural_networks/models/PERSONALITY_PART.h5')

img = io.imread("./temp_photos/3faces.jpg")
faces = crop_rectangle(img)
if faces is not None:
    img = np.array(faces) / 255

    embedding = do_embedding(embedding_model, [img])
    age = predict_age(age_model, embedding)
    gender = predict_gender(gender_model, embedding)
    person_embedding = predict_person(person_model, embedding)
    write_annoy(person_embedding)
    annoy_index = read_annoy(person_embedding[0])

    print("Возраст: ", age)
    print("Пол: ", gender)
    print("Индекс ближайшего в базе: ", annoy_index)
    print("Размерность эмбедингов: ", np.array(person_embedding).shape)

    for im in img:
        plt.imshow(im)
        plt.show()
else:
    print("No faces!")