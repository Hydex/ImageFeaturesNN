from features import *
import pickle
import nn

class CatClassifier:
    def __init__(self):
        self.network = nn.NN(225, 295, 1)

    def classify(self, image_path):
        image = get_image_features(image_path)
        output = self.network.update(image)
        print output
        
        if output[0] > 0.99:
            return "Cat"
        else:
            return "Not cat"

    def preprocess(self, path):
        with open(path, "a+") as f:
            pickle.dump(generate_training_set("trainingdata/cats/*.jpg", 1.0) + generate_training_set("trainingdata/not_cats/*.jpg", 0.0), f)

    def train(self, path):
        with open(path, "r") as f:
            training_data = pickle.load(f)

        self.network.train(training_data, 200)
        

if __name__ == "__main__":
    catc = CatClassifier()
    catc.preprocess("preprocess/cats.dat")
    catc.train("preprocess/cats.dat")

    catc.classify("testing/cat1.jpg")
    catc.classify("testing/dog3.jpg")
