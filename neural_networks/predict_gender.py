from keras.models import load_model


def predict_gender(embedding):
    gender_model = load_model('./neural_networks/models/FEMALE_MALE_PART.h5')
    gender = gender_model.predict(embedding)
    return gender
