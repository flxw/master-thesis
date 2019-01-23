import numpy as np
from math import ceil
from tqdm import trange

def calculate_percentile_precisions(model, input_batches, output_batches, percentile_steps = 10, window_offset = 0):
    percentile_matches  = [0] * percentile_steps
    percentile_elements = [0] * percentile_steps

    for batch_id in range(len(output_batches)):
        batch_x = { layer_name: input_batches[layer_name][batch_id] for layer_name in input_batches.keys() }
        batch_y = output_batches[batch_id]
        
        if len(batch_y) == 0: continue
        
        y_hat = model.predict(batch_x)
        trace_length = batch_x['seq_input'].shape[1] if window_offset == 0 else batch_x['seq_input'].shape[0]
        percentile_step = (trace_length+window_offset) / percentile_steps
        current_percentile = 1 if window_offset == 0 else ceil(window_offset/percentile_step)

        # loop through predicted next steps and compare
        for i in range(0,trace_length):
            if i+window_offset > (current_percentile * percentile_step):
                current_percentile += 1

            # infer one-hot encoding and check if a prediction has been made correctly
            y_batch_hat = y_hat[0,i] if window_offset == 0 else y_hat[i]
            y_batch_target = batch_y[0,i] if window_offset == 0 else batch_y[i]
            
            if np.argmax(y_batch_hat) == np.argmax(y_batch_target):
                # 0-based indexing, this is actually the 1st percentile
                percentile_matches[current_percentile -1] += 1
            percentile_elements[current_percentile -1] += 1

    return [p_m / p_t if p_t != 0 else p_m/1 for p_m,p_t in zip(percentile_matches,percentile_elements)]

def calculate_windowed_precision(model, input_batches, output_batches, window_size, percentile_steps = 10):
    return calculate_percentile_precisions(model, input_batches, output_batches,
                                          percentile_steps, window_size)

def calculate_windowed_precision_with_whole_history(model, input_batches, output_batches, percentile_steps = 10):
    percentile_matches  = [0] * percentile_steps
    percentile_elements = [0] * percentile_steps

    for batch_id in trange(len(output_batches)):
        trace_length = output_batches[batch_id].shape[1]
        if trace_length == 0: continue
        predictions = []

        # successively walk through the trace and collect the predictions for each step
        for i in range(1, trace_length+1):
            batch_x = { 'seq_input': np.array([input_batches['seq_input'][batch_id][0][:i]]) }
            if 'sec_input' in input_batches.keys():
                batch_x['sec_input'] = np.array(input_batches['sec_input'][batch_id][0])
                
            predictions.append(model.predict(batch_x)[0])
            
        percentile_step = (trace_length) / percentile_steps
        current_percentile = 1
        
        # loop through predicted next steps and compare
        for y_hat_idx, y_hat in enumerate(predictions):
            if y_hat_idx > (current_percentile * percentile_step):
                current_percentile += 1

            # infer one-hot encoding and check if a prediction has been made correctly
            y_target = output_batches[batch_id][0][y_hat_idx]
            
            if np.argmax(y_hat) == np.argmax(y_target):
                # 0-based indexing, this is actually the 1st percentile
                percentile_matches[current_percentile -1] += 1
            percentile_elements[current_percentile -1] += 1

    return [p_m / p_t for p_m,p_t in zip(percentile_matches,percentile_elements)]
