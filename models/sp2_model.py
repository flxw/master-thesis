from hyperopt import STATUS_OK
from hyperas import optim
from hyperas.distributions import choice, uniform

def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    from keras.models import Sequential, Model
    from keras.layers import Dense, Embedding, Input, Reshape, concatenate, Flatten, Activation, LSTM

    models = []
    model_inputs = []

    # forward all ordinal features
    for ord_var in feature_names[:cat_col_start_index]:
        il = Input(batch_shape=(1,1), name=generate_input_name(ord_var))
        model = Reshape(target_shape=(1,1,))(il)
        model_inputs.append(il)
        models.append(model)

    # create embedding layers for every categorical feature
    for cat_var in categorical_feature_names :
        model = Sequential()
        no_of_unique_cat  = len(feature_dict[cat_var]['to_int'])
        embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 50 ))
        vocab  = no_of_unique_cat+1

        il = Input(batch_shape=(1,1), name=generate_input_name(cat_var))    
        model = Embedding(vocab, embedding_size)(il)
        model = Reshape(target_shape=(1,embedding_size,))(model)

        model_inputs.append(il)
        models.append(model)

    # create input and embedding for sp2 features
    il = Input(batch_shape=(1,n_sp2_features), name=generate_input_name("sp2"))
    model_inputs.append(il)
    sp2_embedding = il
    # TODO mimic embedding architecture
    # sequence_embedding = Reshape(target_shape=(n_sp2_features,))(sequence_embedding)

    # merge the outputs of the embeddings, and everything that belongs to the most recent activity executions
    main_output = concatenate(models, axis=2)
    main_output = LSTM(25*32, batch_input_shape=(1,), stateful=True)(main_output) # should be multiple of 32 since it trains faster due to np.float32
    main_output = LSTM(25*32, stateful=True)(main_output) # should be multiple of 32 since it trains faster due to np.float32

    # after LSTM has learned on the sequence, bring in the SP2/PFS features, like in Shibatas paper
    main_output = concatenate([main_output, sp2_embedding])
    main_output = Dense(20*32, activation='relu', name='dense_join')(main_output)
    main_output = Dense(len(feature_dict["concept:name"]["to_int"]), activation='sigmoid', name='dense_final')(main_output)

    full_model = Model(inputs=model_inputs, outputs=[main_output])
    full_model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_train, Y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}