import numpy as np

def calculate_percentile_precisions(model, input_batches, output_batches, percentile_steps = 10):
    percentile_matches  = [0] * percentile_steps
    percentile_elements = [0] * percentile_steps

    for batch_id in range(len(output_batches)):
        batch_x = { layer_name: input_batches[layer_name][batch_id] for layer_name in input_batches.keys() }
        batch_y = output_batches[batch_id]
        
        y_hat = model.predict(batch_x)[0]
        current_percentile = 1
        percentile_step = batch_x['seq_input'].shape[1] / percentile_steps

        # loop through predicted next steps and compare
        for i in range(0,batch_y.shape[1]):
            if i > (current_percentile * percentile_step):
                current_percentile += 1

            # infer one-hot encoding and check if a prediction has been made correctly
            if np.argmax(y_hat[i]) == np.argmax(batch_y[0,i]):
                # 0-based indexing, this is actually the 1st percentile
                percentile_matches[current_percentile -1] += 1
            percentile_elements[current_percentile -1] += 1

    return [p_m / p_t for p_m,p_t in zip(percentile_matches,percentile_elements)]
