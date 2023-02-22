import tensorflow as tf
from tensorflow import keras
from keras import models, layers
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import numpy as np
from random import randint

def prep_parameters(setup):
    setup = int(setup)
    if (setup == 1):
        aux = ['B', 'A']
    elif (setup == 2):
        aux = ['B', 'B']
    elif (setup == 3):
        aux = ['B', 'C']
    elif (setup == 4):
        aux = ['L', 'A']
    elif (setup == 5):
        aux = ['L', 'B']
    elif (setup == 6):
        aux = ['L', 'C']
    else:
        print('Opção inválida.')
        return 0
    
    if (aux[0] == 'B'):
        df['G3'].replace([i for i in range(0, 10)], 0, inplace=True)
        df['G3'].replace([i for i in range(10, 21)], 1, inplace=True)
    else:
        df['G3'].replace([i for i in range(0, 10)], 0, inplace=True)
        df['G3'].replace([10, 11], 1, inplace=True)
        df['G3'].replace([12, 13], 2, inplace=True)
        df['G3'].replace([14, 15], 3, inplace=True)
        df['G3'].replace([i for i in range(16, 21)], 4, inplace=True)

    removable_features = list()
    
    if (aux[1] == 'A'):
        if (aux[0] == 'L'):
            epochs = 400
        else:
            epochs = 225

    elif (aux[1] == 'B'):
        removable_features.append('G2')
        if (aux[0] == 'L'):
            epochs = 750
        else:
            epochs = 325

    else:
        removable_features.append('G2')
        removable_features.append('G3')
        if (aux[0] == 'L'):
            epochs = 1500
        else:
            epochs = 375
        
    return [removable_features, epochs]

df1 = pd.read_csv('./student-por.csv', sep=';')
df2 = pd.read_csv('./student-mat.csv', sep=';')
frames = [df1, df2]
df = pd.concat(frames)

print('-'*75)
setup = print("""A previsão podera ser feita utilizando 6 configurações diferentes:
        1 - Classificação Binária, nenhum atributo excluído;
        2 - Classificação Binária, atributo G2 excluído;
        3 - Classificação Binária, atributos G1 e G2 excluídos;
        4 - Classificação em 5 níveis, nenhum atributo excluído;
        5 - Classificação em 5 níveis, atributo G2 excluído;
        6 - Classificação em 5 níveis, atributos G1 e G2 excluídos.
        Informe a configuração desejada: """)
print('-'*75)


setup = input('Informe a configuração desejada: ')
parameters = prep_parameters(setup)


#Substituindo valores de string para valores inteiros nos atributos
df['school'].replace({'GP':0, 'MS':1}, inplace=True)
df['sex'].replace({'M':0, 'F':1}, inplace=True)
df['address'].replace({'U':0, 'R':1}, inplace=True)
df['famsize'].replace({'LE3':0, 'GT3':1}, inplace=True)
df['Pstatus'].replace({'T':0, 'A':1}, inplace=True)
df['Mjob'].replace({'teacher':0, 'health':1, 'services':2, 'at_home':3, 'other':4}, inplace=True)
df['Fjob'].replace({'teacher':0, 'health':1, 'services':2, 'at_home':3, 'other':4}, inplace=True)
df['reason'].replace({'home':0, 'reputation':1, 'course':2, 'other':3}, inplace=True)
df['guardian'].replace({'mother':0, 'father':1, 'other':2}, inplace=True)
df.replace({'no':0, 'yes':1}, inplace=True)
df.drop(columns=parameters[0])

X = df.loc[:, df.columns != 'G3'].values
y = df['G3'].values

#HiperParametros
learning_rate = 0.001
input_dim = len(X[0])
output_num = len(np.unique(df['G3']))
max_epochs = parameters[1]

seed = [randint(0, 10000), randint(0, 10000)]

kf = KFold(10 , shuffle=True, random_state=seed[0])
fold = 0
accuracy_history = list()
my_models = list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed[1])

print(len(X_train))
for train, val in kf.split(X_train, y_train):
    fold += 1

    X_train_k = X_train[train]
    y_train_k = y_train[train]
    X_val_k = X_train[val]
    y_val_k = y_train[val]

    model = models.Sequential(
        [    
            layers.Dense(units=200, input_dim=input_dim, activation='sigmoid'),
            layers.Dense(units=100, activation='sigmoid'),
            layers.Dense(units=50, activation='sigmoid'),
            layers.Dense(units=output_num, activation='softmax')
        ]
    )    

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_k, y_train_k, epochs=max_epochs, validation_data=(X_val_k, y_val_k))

    evaluation = model.evaluate(X_val_k, y_val_k)
    accuracy_history.append(evaluation[1])
    my_models.append(model)

for i in range(fold):
    print(f"Fold {i+1} accuracy: {accuracy_history[i]*100:.4f}%")

print(f'Fold escolhido: {np.argmax(accuracy_history)+1}')
my_model = my_models[np.argmax(accuracy_history)]

test_evaluation = my_model.evaluate(X_test, y_test)

print(f'A taxa de acerto do modelo foi de: {test_evaluation[1]*100:.4f}%')