from sklearn import neural_network
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

    learing_rate_candidates = [0.1, 0.2, 0.3]
    hidden_layer_size = [(32, 16), (20, 16)]
    lambda_candidates = [0.001, 0.002, 0.003]

    # 搜索学习率
    for lr in learing_rate_candidates:
        save_path = "./save/weights_lr_{}.pkl".format(str(lr))
        nn = NeuralNet(28*28, 32, 16, 10, learning_rate=lr)
        nn.train(X, Y, X_test, Y_test, save_path)
        nn.evaluate(X_test, Y_test)
    
    # 搜索正则化强度
    for norm in lambda_candidates:
        save_path = "./save/weights_lambda_{}.pkl".format(str(norm))
        nn = NeuralNet(28*28, 32, 16, 10, norm=norm)
        nn.train(X, Y, X_test, Y_test, save_path)
        nn.evaluate(X_test, Y_test)
    
    # 搜索隐藏层大小
    for one, two in hidden_layer_size:
        save_path = "./save/weights_hidden_{}_{}.pkl".format(str(one), str(two))
        nn = NeuralNet(28*28, one, two, 10)
        nn.train(X, Y, X_test, Y_test, save_path)
        nn.evaluate(X_test, Y_test)
        