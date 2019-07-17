import numpy as np

import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    return parser.parse_args()

def init_normal(shape):
    return K.random_normal(shape, stddev=0.01)

def get_model(layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP

    # Input variables
    user_input = Input(shape=(31,), dtype='float32', name = 'user_input')
    item_input = Input(shape=(55,), dtype='float32', name = 'item_input')

    vector = Concatenate()([user_input, item_input])
    
    # MLP layers
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], kernel_initializer = init_normal ,kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
    return model


def get_train_instances(num_negatives):

    train_users, train_items = dataset.train_users, dataset.train_items
    labels_pos = np.full(len(train_users),1)
    for j in range(num_negatives):
        for i in range(len(dataset.test_users)):
            train_users.append(dataset.test_users[i])
            train_items.append(dataset.testNegatives[i][j])
    labels_neg = np.full((len(dataset.test_users)*num_negatives),0)
    labels = np.concatenate((labels_pos,labels_neg))
    return train_users, train_items, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    verbose = args.verbose
    epochs =  args.epochs
    topK = 10
    evaluation_threads = 1 
    print("MLP arguments: %s " %(args))
    model_out_file = 'Model/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)

    print("Load data done [%.1f s]. "  %(time()-t1))
    # Build model

    model = get_model(layers, reg_layers)

    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')    
    
    # Check Init performance
    t1 = time()
    test_users , test_items , testNegatives = dataset.test_users, dataset.test_items, dataset.testNegatives

    (hits, ndcgs) = evaluate_model(model, test_users, test_items, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f s]' %(hr, ndcg, time()-t1))
    

    # Train model

    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    # Generate training instances
    user_input, item_input ,labels = get_train_instances(num_negatives)

    for epoch in range(10):
        t1 = time()
        # Training        
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True)

        t2 = time()
        
        # Evaluation   
        (hits, ndcgs) = evaluate_model(model, test_users, test_items, testNegatives, topK, evaluation_threads)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                % ((epoch + 1)*epochs ,  t2-t1, hr, ndcg, loss, time()-t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            if args.out > 0:
                model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
