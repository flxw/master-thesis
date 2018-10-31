from hyperopt import STATUS_OK
from hyperas import optim
from hyperas.distributions import choice, uniform

def create_model(train_traces, test_traces):
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
    
    model_inputs = []
    models = []

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

    # create input and embedding for sp2/pfs2 features
    learn_sp2 = True
    sequence_embedding = None

    # Can't embed SP2 due to dimensionality with embedding layer, be stringent and do the same for PFS features
    # instead, mimic the embedding internal architecture and use a Dense/Linear layer
    if learn_sp2:
        il = Input(batch_shape=(1,n_sp2_features), name=generate_input_name("sp2"))
        model_inputs.append(il)
        sequence_embedding = il
        # TODO mimic embedding architecture
        sequence_embedding = Reshape(target_shape=(n_sp2_features,))(sequence_embedding)
    else:
        # TODO
        pass

    # merge the outputs of the embeddings, and everything that belongs to the most recent activity executions
    main_output = concatenate(models, axis=2)
    main_output = LSTM(25*32, batch_input_shape=(1,), return_sequences=True, stateful=True)(main_output) # should be multiple of 32 since it trains faster due to np.float32
    main_output = LSTM(25*32, stateful=True)(main_output) # should be multiple of 32 since it trains faster due to np.float32

    # after LSTM has learned on the sequence, bring in the SP2/PFS features, like in Shibatas paper
    main_output = concatenate([main_output, sequence_embedding], axis=1)
    main_output = Dense(20*32, activation='relu', name='dense_join')(main_output)
    main_output = Dense(len(feature_dict["concept:name"]["to_int"]), activation='sigmoid', name='dense_final')(main_output)

    full_model = Model(inputs=model_inputs, outputs=[main_output])
    full_model.compile(loss='categorical_crossentropy', optimizer={{choice(['adadelta', 'adam', 'sgd'])}}, metrics=['categorical_accuracy', 'mae'])
    
    n_epochs = 40
    for epoch in range(n_epochs):
        mean_tr_acc  = []
        mean_tr_loss = []

        for t in tqdm(train_traces, desc="Epoch {0}/{1}".format(epoch,n_epochs)):
            for x,y in zip(t['x'],t['y']):
                tr_acc, tr_loss = full_model.train_on_batch(x, y)
                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)
            full_model.reset_states()

        print('Epoch {0} -- categorical_acc = {1} -- mae loss = {2}'.format(epoch, np.mean(mean_tr_acc), np.mean(mean_tr_loss)))
        
        if epoch % 5 == 0:
            full_model.save("complete_model_{0}_{1}.h5".format(full_model.optimizer, type(full_model.optimizer).__name__))
            
    npreds = 0
    correct_preds = 0
    for t in test_traces[0:1]:
        for x,y in zip(t['x'],t['y']):
            npreds += 1
            pred_y = full_model.predict(x)
            correct_preds += pred_y == y
        full_model.reset_states()
        
    return {'loss': -1 * correct_preds/npreds, 'status': STATUS_OK, 'model': model}