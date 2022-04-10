from Prepare_Data import *
from Neural_Network import *


if __name__ == '__main__':

    load_data = Prepare_Data()
    load_data.read_files()
    test = load_data.get_test
    train = load_data.get_training

    X = train['train_images'].reshape(train['train_images'].shape[0], 28*28) / 255.
    X_test = test['test_images'].reshape(test['test_images'].shape[0], 28*28) / 255.

    Y = train['train_labels']
    Y_test = test['test_labels']
    save_path = "./save/weights.pkl"

    nn = NeuralNet(28*28, 32, 16, 10)
    # nn.load(save_path)
    nn.train(X, Y, X_test, Y_test, save_path)
    nn.evaluate(X_test, Y_test)
