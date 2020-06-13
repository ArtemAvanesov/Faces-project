from keras.models import load_model


def predict_person(embedding):
    person_model = load_model('./neural_networks/models/PERSONALITY_PART.h5')
    person_embedding = person_model.predict(embedding)
    return person_embedding
