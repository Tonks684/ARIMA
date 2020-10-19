from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense


def base_model_without_drpout(timesteps,numberoffeatures,stepsout,epoch_num,train_x,train_y,batch_size):
    
    model = Sequential()
    model.add(LSTM(100, activation = "relu", return_sequences=True, input_shape=(timesteps, numberoffeatures)))   
    model.add(LSTM(100, activation = "relu", return_sequences=True, input_shape=(timesteps, numberoffeatures)))   
    model.add(LSTM(100, activation = "relu", return_sequences=False, input_shape=(timesteps, numberoffeatures)))   
    #model.add(LSTM(100, activation = "relu", kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(stepsout)) #removed initialisers
    model.compile(loss="mae", optimizer='adam', metrics= ["mae"])
    model.summary()
    model.fit(train_x,train_y, epochs=epoch_num, batch_size=batch_size)
    
    return model
