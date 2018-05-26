from sklearn.metrics import classification_report
import argparse
from utils import load_images_to_memory, get_inception_with_frozen_layers

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
    help="path to model")
args = vars(ap.parse_args())

le, data, labels = load_images_to_memory(args["dataset"])

model = get_inception_with_frozen_layers(102)

model.load_weights(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(data, batch_size=64)
print(classification_report(labels.argmax(axis=1),
    predictions.argmax(axis=1), target_names=le.classes_))

