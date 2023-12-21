import numpy as np
import pandas as pd

from .idsw.datafetch.core import InvalidInputsError
from .idsw.etl.strings import switch_strings
from .idsw.etl.transform import (get_frequency_features, feature_scaling)

from .core import GlobalVars

from .models import (
  calculate_leading_current_power_factor,
  create_clusters,
  prediction_pipeline
)

def check_inputs(dataset):
  """Check if all inputs are within the valid ranges.
  If they are, apply linear correlation to calculate the
  leading current power factor"""
  
  possible_ranges = GlobalVars.possible_ranges
  checked_variables = ['lagging_current_reactive_power', 'leading_current_reactive_power',
                      'co2_tco2', 'lagging_current_power_factor']
  
  total_entries = len(dataset) # total of entries to check

  for var in checked_variables:
    # Check if the variables are within the valid range. If they are not, raise an error:
    var_min = possible_ranges[var]['min']
    var_max = possible_ranges[var]['max']

    # Create an array containing only the analyzed variable:
    var_array = np.array(dataset[var])
    """
    Suppose var_array = np.array([1, 2, 3, 4])
    if var_min = 2
    (var_array >= 2) will return array([False,  True,  True,  True])
    False is automatically converted to 0; True is converted to 1. Thus,
    np.sum(var_array >= 2) returns 3.
    Notice that 3 < len(var_array), which is 4. So, if the sum is different
    than the length, the condition is not verified for all elements, so the error must
    be raised.
    """
    if (np.sum(var_array >= var_min) != total_entries):
      raise InvalidInputsError(f"The minimum value allowed for variable {var} is {var_min}, but the simulation found at least one value lower than this.")
    
    elif (np.sum(var_array <= var_max) != total_entries):
      raise InvalidInputsError(f"The maximum value allowed for variable {var} is {var_max}, but the simulation found at least one value higher than this.")
    
    else:
      pass
  
  # Now that the simulation passed through the checking phase, apply the linear correlation:
  leading_current_reactive_power = np.array(dataset['leading_current_reactive_power'])
  leading_current_power_factor = calculate_leading_current_power_factor(leading_current_reactive_power)
  # Add to the dataset:
  dataset['leading_current_power_factor'] = leading_current_power_factor

  return dataset


def encode_weekdays(dataset):
  """Encode the weekdays from the dataframe."""

 DATASET = dataset
 COLUMN_TO_ANALYZE = 'day_of_week'
 LIST_OF_DICTIONARIES_WITH_ORIGINAL_STRINGS_AND_REPLACEMENTS = [
      
      {'original_string': 'Monday', 'new_string': '1'}, 
      {'original_string': 'Tuesday', 'new_string': '2'}, 
      {'original_string': 'Wednesday', 'new_string': '3'}, 
      {'original_string': 'Thursday', 'new_string': '4'}, 
      {'original_string': 'Friday', 'new_string': '5'}, 
      {'original_string': 'Saturday', 'new_string': '6'}, 
      {'original_string': 'Sunday', 'new_string': '7'}, 
      
  ]
  CREATE_NEW_COLUMN = False
  NEW_COLUMN_SUFFIX = '_stringReplaced'
  dataset = switch_strings (df = DATASET, column_to_analyze = COLUMN_TO_ANALYZE, list_of_dictionaries_with_original_strings_and_replacements = LIST_OF_DICTIONARIES_WITH_ORIGINAL_STRINGS_AND_REPLACEMENTS, create_new_column = CREATE_NEW_COLUMN, new_column_suffix = NEW_COLUMN_SUFFIX)
  
  dataset['day_of_week'] = dataset['day_of_week'].astype('int')

  return dataset


def encode_weekstatus(dataset):
  """Encode the weekstatus from the dataframe."""

 DATASET = dataset
 COLUMN_TO_ANALYZE = 'weekstatus'
 LIST_OF_DICTIONARIES_WITH_ORIGINAL_STRINGS_AND_REPLACEMENTS = [
      
    {'original_string': 'Weekday', 'new_string': '1'}, 
    {'original_string': 'Weekend', 'new_string': '0'}, 
      
  ]

  CREATE_NEW_COLUMN = False
  NEW_COLUMN_SUFFIX = '_stringReplaced'
  dataset = switch_strings (df = DATASET, column_to_analyze = COLUMN_TO_ANALYZE, list_of_dictionaries_with_original_strings_and_replacements = LIST_OF_DICTIONARIES_WITH_ORIGINAL_STRINGS_AND_REPLACEMENTS, create_new_column = CREATE_NEW_COLUMN, new_column_suffix = NEW_COLUMN_SUFFIX)
  
  dataset['weekstatus'] = dataset['weekstatus'].astype('int')

  return dataset


def add_frequencies(dataset):
  """Add frequency features correspondent to the simulated datetimes."""

  DATASET = dataset
  TIMESTAMP_TAG_COLUMN = "timestamp"
  IMPORTANT_FREQUENCIES = [{'value': 4.002766, 'unit': 'year'}, 
                          {'value': 52.035958, 'unit': 'year'},
                          {'value': 365.252400, 'unit': 'year'},
                          {'value': 1095.757200, 'unit': 'year'},
                          {'value': 1461.009600, 'unit': 'year'},
                          {'value': 1826.262000, 'unit': 'year'},
                          ]

  X_AXIS_ROTATION = 70
  Y_AXIS_ROTATION = 0
  GRID = True #Alternatively: True or False
  HORIZONTAL_AXIS_TITLE = None #Alternatively: string inside quotes for horizontal title
  VERTICAL_AXIS_TITLE = None #Alternatively: string inside quotes for vertical title
  PLOT_TITLE = None #Alternatively: string inside quotes for graphic title
  MAX_NUMBER_OF_ENTRIES_TO_PLOT = None
  EXPORT_PNG = False
  DIRECTORY_TO_SAVE = None
  FILE_NAME = None
  PNG_RESOLUTION_DPI = 330

  dataset, timestamp_dict = get_frequency_features (df = DATASET, timestamp_tag_column = TIMESTAMP_TAG_COLUMN, important_frequencies = IMPORTANT_FREQUENCIES, x_axis_rotation = X_AXIS_ROTATION, y_axis_rotation = Y_AXIS_ROTATION, grid = GRID, horizontal_axis_title = HORIZONTAL_AXIS_TITLE, vertical_axis_title = VERTICAL_AXIS_TITLE, plot_title = PLOT_TITLE, max_number_of_entries_to_plot = MAX_NUMBER_OF_ENTRIES_TO_PLOT, export_png = EXPORT_PNG, directory_to_save = DIRECTORY_TO_SAVE, file_name = FILE_NAME, png_resolution_dpi = PNG_RESOLUTION_DPI)

  dataset = dataset.rename(columns = {
       '0.24982724446045557_year_sin':'freq1_sin', '0.24982724446045557_year_cos': 'freq1_cos',
       '0.019217480343111968_year_sin':'freq2_sin', '0.019217480343111968_year_cos': 'freq2_cos',
       '0.0027378327972656714_year_sin':'freq3_sin', '0.0027378327972656714_year_cos': 'freq3_cos',
       '0.0009126109324218906_year_sin':'freq4_sin', '0.0009126109324218906_year_cos': 'freq4_cos',
       '0.0006844581993164178_year_sin':'freq5_sin', '0.0006844581993164178_year_cos': 'freq5_cos',
       '0.0005475665594531343_year_sin':'freq6_sin', '0.0005475665594531343_year_cos': 'freq6_cos'
  })

  return dataset


def encode_loadtype(dataset):
  """Perform One-Hot Encoding from variable load_type."""

  df = dataset.copy(deep = True)
  load_array = np.array(df['load_type'])

  # Create arrays containing only zeros, with same dimension as load_array:
  light = np.zeros(load_array.shape)
  medium = np.zeros(load_array.shape)
  maximum = np.zeros(load_array.shape)

  # Now, fill the arrays:
  light = np.where((load_array == 'Light_Load'), 1, light) # fill with 1 for light load
  medium = np.where((load_array == 'Medium_Load'), 1, medium)
  maximum = np.where((load_array == 'Maximum_Load'), 1, maximum)

  # Now, create the encoded columns:
  df['load_type_Light_Load_OneHotEnc'] = light
  df['load_type_Maximum_Load_OneHotEnc'] = maximum
  df['load_type_Medium_Load_OneHotEnc'] = medium

  return df


def scale_features(dataset):
  """Perform standard scaling of the features."""

  DATASET = dataset
  SUBSET_OF_FEATURES_TO_SCALE = ['lagging_current_reactive_power_kvarh',
        'leading_current_reactive_power_kvarh', 'co2_tco2', 'usage_kwh']

  MODE = 'standard'
  SCALE_WITH_NEW_PARAMS = False
  LIST_OF_SCALING_PARAMS = [{'column': 'lagging_current_reactive_power_kvarh',
                'scaler': {'scaler_obj': None,
                'scaler_details': {'mu': 13.035383561643835, 'sigma': 14.524747793581406}}},
              {'column': 'leading_current_reactive_power_kvarh',
                'scaler': {'scaler_obj': None,
                'scaler_details': {'mu': 3.8709486301369855, 'sigma': 6.729335287688414}}},
              {'column': 'co2_tco2',
                'scaler': {'scaler_obj': None,
                'scaler_details': {'mu': 0.01152425799086758,
                  'sigma': 0.015072620173269598}}},]
  
  SUFFIX = '_scaled'
  dataset, scaling_list = feature_scaling (df = DATASET, subset_of_features_to_scale = SUBSET_OF_FEATURES_TO_SCALE, mode = MODE, scale_with_new_params = SCALE_WITH_NEW_PARAMS, list_of_scaling_params = LIST_OF_SCALING_PARAMS, suffix = SUFFIX)

  return dataset


def simulation_pipeline(dataset, kmeans_model, encoder_decoder_tf_model):
  """Run the full data transformation and simulation pipeline."""

  # Create copy to manipulate without risks of losing data:
  df = dataset.copy(deep = True)
  # Run the functions:
  df = check_inputs(df)
  
  # Now, get the dataframe that will be used for feeding the model:
  model_df = create_clusters(kmeans_model, df)
  model_df = encode_weekdays(model_df)
  model_df = encode_weekstatus(model_df)
  model_df = add_frequencies(model_df)
  model_df = encode_loadtype(model_df)
  model_df = scale_features(model_df)
  # Predict model output and re-scale it to kWh:
  df = prediction_pipeline(encoder_decoder_tf_model, model_df, df)
  
  columns = ['timestamp',	'lagging_current_reactive_power_kvarh',	'leading_current_reactive_power_kvarh',	
            'co2_tco2',	'lagging_current_power_factor',	'leading_current_power_factor', 'nsm',
            'weekstatus', 'day_of_week', 'load_type', 'usage_kwh']
  
  # Filter and re-order dataframe columns:
  df = df[columns]

  return df
