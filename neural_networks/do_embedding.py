from keras.models import load_model


def do_embedding(face):
    embedding_model = load_model('./neural_networks/models/BASE_MODEL.h5')
    embedding = embedding_model.predict(face)
    return embedding
