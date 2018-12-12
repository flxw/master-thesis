from abc import abc, abstractmethod

class AbstractFormatter(object):
  @abstractmethod @staticmethod
  def format_datasets(model_formatted_data_fn, datapath, target_variable):
    """
    :param model_formatted_data_fn: A callback to the format data function of the current model.
    :type model_formatted_data_fn: function
    :param datapath: The name of the target variable, most often concept:name.
    :type datapath: string
    :returns: Data formatted to the intake requirements of the model as batches.
              Returns are in this order: train_inputs, train_targets, test_inputs, test_targets.
              All inputs are dictionaries, keyed with the name of the input layer.
              The values are arrays of batches of the input data in the required dimension, most often 3D for LSTM inputs
              The targets are also arrays of batches, with the respective dimension of the output, 1D for non-timeseries output, 2D for timeseries output
    :rtype: object
    """
    pass
