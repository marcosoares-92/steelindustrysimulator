import numpy as np
import pandas as pd
import tensorflow as tf

from .idsw.datafetch.pipes import import_export_model_list_dict
from .idsw.modelling.preparetensors import separate_and_prepare_features_and_responses
from .idsw.modelling.utils import make_model_predictions

from .utils import (random_noise, correct_vals_out_of_bounds)


def load_models():
  """Load K-Means clustering and the TensorFlow encoder-decoder model.
  Warning: this function will only work if the sequence of commands in the function
  start_simulation() (__init__ module) properly run, assuring that the directories are
  saved in the correct path.

  Since Colab may impose user restrictions regarding to move or copy files, we could have
  problems on the step of decompressing the model. Thus, we copy the TensorFlow module
  directly from the GitHub repository and do not use IDSW function to load it.
  Instead, we apply only the specific part used for loading a tf model.

  Notice that the model is saved on the GitHub subdirectory 'encoder_decoder_tf_model/saved_model'
  """

  # Move the directory with TensorFlow model to the root directory.
  # If an error occurs during the decompression, the saved model in the folder will be loaded.
  """
  With appropriate permissions, user could move decompressed files using this:
  
  import os
  import shutil
  src = 'steelindustrysimulator/digitaltwin/data/tmp/'
  dst = 'tmp'
  os.makedirs("tmp/", exist_ok = True)
  shutil.copytree(src, dst)
  """
  
  # Shared variables
  ACTION = 'import'
  OBJECTS_MANIPULATED = 'model_only'
  DICTIONARY_OR_LIST_FILE_NAME = None
  DIRECTORY_PATH = 'steelindustrysimulator/digitaltwin/data'
  DICTIONARY_OR_LIST_FILE_NAME = None
  DICT_OR_LIST_TO_EXPORT = None
  MODEL_TO_EXPORT = None 
  USE_COLAB_MEMORY = False

  # Kmeans cluster:
  MODEL_FILE_NAME = 'kmeans_model'  
  MODEL_TYPE = 'sklearn'
  kmeans_model = import_export_model_list_dict (action = ACTION, objects_manipulated = OBJECTS_MANIPULATED, model_file_name = MODEL_FILE_NAME, dictionary_or_list_file_name = DICTIONARY_OR_LIST_FILE_NAME, directory_path = DIRECTORY_PATH, model_type = MODEL_TYPE, dict_or_list_to_export = DICT_OR_LIST_TO_EXPORT, model_to_export = MODEL_TO_EXPORT, use_colab_memory = USE_COLAB_MEMORY) 

  # Deep learning encoder-decoder model:
  model_path = "steelindustrysimulator/digitaltwin/data/encoder_decoder_tf_model/saved_model"
  encoder_decoder_tf_model = tf.keras.models.load_model(model_path)
  print(f"Keras/TensorFlow model successfully imported from {model_path}.")

  return kmeans_model, encoder_decoder_tf_model


def calculate_leading_current_power_factor(leading_current_reactive_power, possible_ranges):
  """Apply the internal linear correlation:
      'Leading_Current_Reactive_Power_kVarh_mean'
      Linear regression summary for Leading_Current_Power_Factor_mean:

      'y = -0.23*x + 23.09'
      'R²_lin_reg = 0.9071'
  """
  leading_current_reactive_power = np.array(leading_current_reactive_power)
  leading_current_power_factor = leading_current_reactive_power*(-0.23) + 23.09

  # Add a random noise to this feature:
  std = possible_ranges['leading_current_power_factor']['std']
  leading_current_power_factor = random_noise(leading_current_power_factor, std)

  # Check if array contains a value above the max or below the minimum.
  var_max = possible_ranges['leading_current_power_factor']['max']
  var_min = possible_ranges['leading_current_power_factor']['min']
  leading_current_power_factor = correct_vals_out_of_bounds(leading_current_power_factor, var_min, var_max)
  
  return leading_current_power_factor


def create_clusters(kmeans_model, dataset):
  """Associate the electric state to one of the clusters"""
  # Create a deep copy for not losing data:
  df = dataset.copy(deep = True)
  DATASET = dataset  #Alternatively: object containing the dataset to be analyzed

  FEATURES_COLUMNS = ['lagging_current_reactive_power',
                    'leading_current_reactive_power',
                    'lagging_current_power_factor',
                    'leading_current_power_factor']
  
  RESPONSE_COLUMNS = 'usage_kwh'
  X, y, column_map_dict = separate_and_prepare_features_and_responses (df = DATASET, features_columns = FEATURES_COLUMNS, response_columns = RESPONSE_COLUMNS)

  MODEL_OBJECT = kmeans_model
  X_tensor = X
  DATAFRAME_FOR_CONCATENATING_PREDICTIONS = dataset  
  COLUMN_WITH_PREDICTIONS_SUFFIX = None
  FUNCTION_USED_FOR_FITTING_DL_MODEL = 'get_deep_learning_tf_model'
  ARCHITECTURE = None
  LIST_OF_RESPONSES = RESPONSE_COLUMNS
  
  dataset = make_model_predictions (model_object = MODEL_OBJECT, X = X_tensor, dataframe_for_concatenating_predictions = DATAFRAME_FOR_CONCATENATING_PREDICTIONS, column_with_predictions_suffix = COLUMN_WITH_PREDICTIONS_SUFFIX, function_used_for_fitting_dl_model = FUNCTION_USED_FOR_FITTING_DL_MODEL, architecture = ARCHITECTURE, list_of_responses = LIST_OF_RESPONSES)
  dataset = dataset.rename(columns = {"y_pred_usage_kwh":"electric_cluster"})

  return dataset


def get_tensor_for_simulation(dataset):
  """Select the features and prepare tensors for running the simulation"""

  DATASET = dataset 
  FEATURES_COLUMNS = ['lagging_current_reactive_power_kvarh_scaled',
        'leading_current_reactive_power_kvarh_scaled', 'co2_tco2_scaled',
        'weekstatus', 'day_of_week', 'load_type_Light_Load_OneHotEnc',
        'load_type_Maximum_Load_OneHotEnc', 'load_type_Medium_Load_OneHotEnc',
        'freq1_sin', 'freq1_cos', 'freq2_sin', 'freq2_cos', 'freq3_sin',
        'freq3_cos', 'freq4_sin', 'freq4_cos', 'freq5_sin', 'freq5_cos',
        'freq6_sin', 'freq6_cos', 'electric_cluster']

  RESPONSE_COLUMNS = 'usage_kwh_scaled'
  X, y, column_map_dict = separate_and_prepare_features_and_responses (df = DATASET, features_columns = FEATURES_COLUMNS, response_columns = RESPONSE_COLUMNS)

  return X, RESPONSE_COLUMNS


def get_model_predictions(encoder_decoder_tf_model, X, RESPONSE_COLUMNS, model_df):
  """Calculate model predictions for the tensor. Notice that these responses
  are still scaled, since model was trained using the standard scaled response."""
  MODEL_OBJECT = encoder_decoder_tf_model
  X_tensor = X
  DATAFRAME_FOR_CONCATENATING_PREDICTIONS = model_df  
  COLUMN_WITH_PREDICTIONS_SUFFIX = None
  FUNCTION_USED_FOR_FITTING_DL_MODEL = 'get_deep_learning_tf_model'
  ARCHITECTURE = 'encoder_decoder'
  LIST_OF_RESPONSES = RESPONSE_COLUMNS
  
  model_df = make_model_predictions (model_object = MODEL_OBJECT, X = X_tensor, dataframe_for_concatenating_predictions = DATAFRAME_FOR_CONCATENATING_PREDICTIONS, column_with_predictions_suffix = COLUMN_WITH_PREDICTIONS_SUFFIX, function_used_for_fitting_dl_model = FUNCTION_USED_FOR_FITTING_DL_MODEL, architecture = ARCHITECTURE, list_of_responses = LIST_OF_RESPONSES)
  model_df = model_df.rename(columns = {'y_pred_usage_kwh_scaled': 'usage_kwh_scaled'})

  return model_df


def rescale_response(predicted_values):
  """Pick an array with predicted responses and reconvert to the original kwh scale.
  : param: predicted_values contain the y_pred model predictions for the inputs.
  
  scaled_var = (var - mu)/sigma
  var = scaled_var*sigma + mu

  {'column': 'usage_kwh',
    'scaler_details': {'mu': 27.386892408675802, 'sigma': 31.352646806775816}}}
  """

  usage_kwh_arr = (np.array(predicted_values))*(31.352646806775816) + (27.386892408675802)

  return usage_kwh_arr


def prediction_pipeline(encoder_decoder_tf_model, model_df, df):
  """Run full pipeline of preparing tensors, getting the model predictions and reconverting it
      to the appropriate kWh scale"""

  X, RESPONSE_COLUMNS = get_tensor_for_simulation(model_df)
  model_df = get_model_predictions(encoder_decoder_tf_model, X, RESPONSE_COLUMNS, model_df)
  scaled_predictions = model_df['y_pred_usage_kwh_scaled']
  model_predictions = rescale_response(predicted_values)
  # Add the predictions to the correct dataset:
  df['usage_kwh'] = model_predictions

  return df
