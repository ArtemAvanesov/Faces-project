from keras.models import load_model


def predict_age(embedding):
    age_model = load_model('./neural_networks/models/AGE_PART.h5')
    age = age_model.predict(embedding)
    return age
