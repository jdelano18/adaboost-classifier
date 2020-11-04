from utils import readArff, preprocess_data, accuracy_score, parse_filename
from AdaBoost import AdaBoost

if __name__ == '__main__':
    filename = parse_filename()

    print("\nUsing entire dataset for train and test")
    X,y = preprocess_data(readArff(filename))
    ada = AdaBoost()
    ada.train(X,y)
    preds = ada.predict(X)
    acc = accuracy_score(preds, y)
    print("Accuracy:", acc)
