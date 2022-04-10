from cmath import exp
import random
from tkinter import HIDDEN
from unittest import result
import numpy as np
import scipy.special as sci
import os
from torch import batch_norm
from tqdm import tqdm
import math
import pickle
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1, norm=0.001):

        self.inputs = None
        self.answers = None
        self.lr = learning_rate

        self.input_size = input_size
        self.hidden_layer_size1 = hidden_size1
        self.hidden_layer_size2 = hidden_size2
        self.output_layer_size = output_size

        self.hidden_weights_1 = 2 * np.random.random((self.input_size, self.hidden_layer_size1)) - 1
        self.hidden_weights_2 = 2 * np.random.random((self.hidden_layer_size1, self.hidden_layer_size2)) - 1
        self.output_weights = 2 * np.random.random((self.hidden_layer_size2, self.output_layer_size)) - 1

        self.correct_output = None
        self.total_error = None

        self.layer_z1 = None
        self.layer_1_output = None
        self.layer_1_error = None
        self.layer_1_delta = None

        self.layer_z2 = None
        self.layer_2_output = None
        self.layer_2_error = None
        self.layer_2_delta = None

        self.output = None
        self.output_error = None
        self.output_delta = None

        self.loss = []
        self.epoch = []
        self.acc = []

        self.epochs = 50
        self.batch_size = 32
        self.decay_rate = 0.1
        self.norm = norm

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return sci.expit(x) * (1 - sci.expit(x))
            #return x * (1-x)

        return sci.expit(x)

    @staticmethod
    def net_output(x):
        max_val = 0
        pos = None
        for i in range(x.shape[0]):
            if x[i] > max_val:
                max_val = x[i]
                pos = i
        answer = np.zeros(10, dtype=np.float)
        answer[pos] = 1
        return answer

    def forward_propagate(self, X):
        self.layer_z1 = np.dot(X, self.hidden_weights_1)
        self.layer_1_output = self.sigmoid(self.layer_z1)
        self.layer_z2 = np.dot(self.layer_1_output, self.hidden_weights_2)
        self.layer_2_output = self.sigmoid(self.layer_z2)
        self.output = self.sigmoid(np.dot(self.layer_2_output, self.output_weights))

    def calculate_error(self):
        self.output_error = (self.correct_output - self.output)
        self.total_error = np.mean(np.abs(self.output_error))

    def back_propagate(self, input, answer):
        self.correct_output = answer
        self.forward_propagate(input.copy())
        self.calculate_error()

        # calculate output delta
        self.output_delta = self.output_error * self.sigmoid(self.output, derivative=True)

        # Calculate hidden layer 2 error
        self.layer_2_error = np.dot(self.output_delta, np.transpose(self.output_weights))

        # Calculate hidden layer 2 delta
        self.layer_2_delta = self.layer_2_error * self.sigmoid(self.layer_z2, derivative=True)

        # Calculate hidden layer 1 error
        self.layer_1_error = np.dot(self.layer_2_delta, np.transpose(self.hidden_weights_2))

        # calculate hidden layer 1 delta
        self.layer_1_delta = self.layer_1_error * self.sigmoid(self.layer_z1, derivative=True)

        # update weights using calculated deltas
        self.output_weights += self.lr * (np.dot(np.transpose(self.layer_2_output), self.output_delta) - self.norm * self.output_weights)
        self.hidden_weights_2 += self.lr * (np.dot(np.transpose(self.layer_1_output), self.layer_2_delta) - self.norm * self.hidden_weights_2)
        self.hidden_weights_1 += self.lr * (np.dot(np.transpose(input), self.layer_1_delta) - self.norm * self.hidden_weights_1)

    def train(self, inputs, outputs, X_test, Y_test, save_path):
        self.inputs = inputs
        self.answers = outputs
        
        input_size = len(self.inputs)
        batchnum = int(input_size / self.batch_size)
        index = list(range(input_size))
        acc = 0.0

        for epoch in range(self.epochs):
            self.epoch.append(epoch+1)
            random.shuffle(index)
            total_loss = 0.0 
            mean_loss = 0.0
            self.lr = self.lr * math.exp(-self.decay_rate * epoch)
            with tqdm(range(batchnum)) as t:
                t.set_description("Epoch %i" % epoch)
                for k in t:
                    self.back_propagate(self.inputs[index[k*self.batch_size:(k+1)*self.batch_size]], 
                                        self.answers[index[k*self.batch_size:(k+1)*self.batch_size]])
                    total_loss += np.mean(np.abs(self.output_error))
                    mean_loss = total_loss/(k+1)
                    t.set_postfix(loss=mean_loss, lr=self.lr)
            res = self.evaluate(X_test, Y_test)
            self.acc.append(res)
            self.loss.append(mean_loss)
            if res > acc:
                acc = res
                self.save(save_path)
            elif epoch >= 30:
                print("Early Stop")
                break
        
        # self.loss_plot()
        self.acc_plot()

    def save_training(self, dir, file_name):
        if not os.path.exists(dir):
            os.makedirs(dir)

        if os.path.exists(dir + file_name + '.npz'):
            print(file_name + " already exists\n")
            print("Please enter a new file name:    ")
            file_name = input()

        save_path = dir + '/' + file_name
        np.savez(save_path, hidden=self.hidden_weights_1, output=self.output_weights)

    def load_training(self, dir, file_name):
        file_path = dir + file_name + '.npz'
        if not os.path.exists(file_path):
            print("This file does not exist.")
            return
        saved = np.load(file_path)
        self.hidden_weights_1 = saved['hidden']
        self.output_weights = saved['output']
        saved.close()
        return

    def evaluate(self, inputs, answers):
        self.forward_propagate(inputs)
        assert self.output.shape == (len(inputs), 10)
        result = 0
        for i in range(len(inputs)):
            predict = self.net_output(self.output[i])
            judge = (predict == answers[i])
            if judge.all():
                result += 1
        acc = result / len(inputs)
        print("Acc = {}".format(acc))
        return acc
    
    def save(self, save_path):
        weights = {"layer1_weight": self.hidden_weights_1,
                   "layer2_weight": self.hidden_weights_2,
                   "output_weight": self.output_weights}
        with open(save_path, "wb") as f:
            pickle.dump(weights, f)
    
    def load(self, save_path):
        with open(save_path, "rb") as f:
            weights = pickle.load(f)
            self.hidden_weights_1 = weights["layer1_weight"]
            self.hidden_weights_2 = weights["layer2_weight"]
            self.output_weights = weights["output_weight"]
    
    def loss_plot(self):
        plt.plot(self.epoch, self.loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def acc_plot(self):
        plt.plot(self.epoch, self.acc)
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.show()

