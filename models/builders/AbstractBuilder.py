from abc import ABC, abstractmethod

class AbstractBuilder(ABC):
  @abstractmethod @staticmethod
  def prepare_datasets(path_to_original_data, target_variable):
    """
    :param path_to_original_data: Path to the folder where the pickled, prepared data resides.
    :type n_train_cols: string
    :param target_variable: The name of the target variable, most often concept:name.
    :type target_variable: string
    :returns: A tupel with of train_X, train_Y, test_X, test_Y.
              X returns are dictionaries of arrays, keyed with the appropriate layer name.
              Y returns are arrays. All arrays contain one trace, which in turn contains one array for each timestep.
    :rtype: Tupel
    """
    pass

  @abstractmethod @staticmethod
  def construct_model(n_train_cols, n_target_cols, learn_windows=False):
    """
    :param n_train_cols: A dictionary of the number of input columns, keyed with the respective layer name
    :type n_train_cols: Tuple
    :param n_target_cols: number of target columns in the output layer
    :type n_target_cols: int
    :returns: Keras model of the constructed model
    :rtype: object
    """
    pass
