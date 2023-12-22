import random
from datetime import datetime
import numpy as np
import pandas as pd

from .models import calculate_leading_current_power_factor


def random_noise(array, std):
  """Create a random noise with uniform distribution to add to variables.
  : param: array: np.ndarray to which the noise will be added.
  : param: std: standard deviation of the original variable. 
  The noise is generated between 0 and 1 with function
  numpy.random.rand.
  https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html

  Then, we multiply this value by 3 times the standard deviation to simulate the actuav
  variability.
  
  1. Use np.random.choice(arr, number_of_samples, p = list_of_probabilities) function.
  https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
  We set arr = [-1, 1], p = [0.5, 0.5] to create an array of -1 and 1 with equal probabilities.
  arr must be with same size as the input array.
  2. We create a second array with random elements.
  3. Then, we use np.multiply function to multiply element-wisely each element from the arrays,
  so that noises can be positive or negative.
  Note: np.multiply: element-wise multiplications; np.matmul: matrix multiplication with rules
  from linear algebra; np.dot: dot product of the vectors.
  https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
  https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
  """

  total_elements = len(array)
  # 1. array of -1 and 1:
  pos_or_neg = np.random.choice([1, -1], total_elements, p = [0.5, 0.5])
  # If the array has 5 elements, pos_or_neg will be like array([-1, -1, -1, -1,  1])
  # 2. noise: firstly, create a random array and multiply by 3x std
  noise_arr = (np.random.rand(total_elements)) * 3 * std
  # 3. Apply np.multiply to get positive and negative noises from 0 to 3*std:
  noise_arr = np.multiply(noise_arr, pos_or_neg)
  # 4. Finally, add the random noise to the array:
  array = array + noise_arr

  return array


def correct_vals_out_of_bounds(array, var_min, var_max):
  """The random noise can generate arrays with values above
  the maximum of the variable in the training dataset, var_max,
  or below the minimum var_min (the boundaries).
  This function replaces values higher than var_max by var_max,
  and values lower than var_min by var_min, correcting values out of boundaries.
  """

  # Replace values above var_max by var_max
  array = np.where(array > var_max, var_max, array)
  # Replace values below var_min by var_min
  array = np.where(array < var_min, var_min, array)

  return array


def load_df_and_ranges():
  """Load original dataframe used for modelling and ranges allowed for each input variable
  Warning: this function will only work if the sequence of commands in the function
  start_simulation() (__init__ module) properly run, assuring that the directories are
  saved in the correct path.
  """
  
  # Read the Pandas dataframe used for training for retrieving operation ranges:
  df = pd.read_csv('steelindustrysimulator/digitaltwin/data/raw_data_by_hour.csv')

  possible_ranges = {

    'lagging_current_reactive_power_kvarh':{'min': df['Lagging_Current_Reactive.Power_kVarh_mean'].min(), 'max':df['Lagging_Current_Reactive.Power_kVarh_mean'].max(), 'std':df['Lagging_Current_Reactive.Power_kVarh_mean'].std()},
    'leading_current_reactive_power_kvarh':{'min': df['Leading_Current_Reactive_Power_kVarh_mean'].min(), 'max':df['Leading_Current_Reactive_Power_kVarh_mean'].max(), 'std':df['Leading_Current_Reactive_Power_kVarh_mean'].std()},
    'co2_tco2':{'min': df['CO2(tCO2)_mean'].min(), 'max':df['CO2(tCO2)_mean'].max(), 'std':df['CO2(tCO2)_mean'].std()},
    'lagging_current_power_factor':{'min': df['Lagging_Current_Power_Factor_mean'].min(), 'max':df['Lagging_Current_Power_Factor_mean'].max(), 'std':df['Lagging_Current_Power_Factor_mean'].std()},
    'leading_current_power_factor':{'min': df['Leading_Current_Power_Factor_mean'].min(), 'max':df['Leading_Current_Power_Factor_mean'].max(), 'std':df['Leading_Current_Power_Factor_mean'].std()}
  
  }

  return df, possible_ranges


def random_start(df):
  """Define random values for starting the input parameters.
     These values are obtained by searching for a random entry on the base-dataset.
     
    - Function random.sample(input_sequence, number_of_samples): 
      this function creates a list containing a total of elements equals to the parameter 
      "number_of_samples", which must be an integer.
      This list is obtained by ramdomly selecting a total of "number_of_samples" elements from the
      list "input_sequence" passed as parameter.
        
    - Function random.choices(input_sequence, k = number_of_samples):
      similarly, randomly select k elements from the sequence input_sequence. This function is
      newer than random.sample
  """

  input_sequence = range(0, len(df)) # total elements from the dataframe: sequence with all possible indices
  # Now, select a random index in input_sequence (select a single sample, i.e., k = 1):
  random_idx = random.choices(input_sequence, k = 1)
  # random_idx is a list containing a single element, e.g. [2]. Since there is a single element,
  # its index is 0
  random_idx = int(random_idx[0])
  # now, get a sample of the dataframe, containing only the row selected:
  df_sample = df.loc[[random_idx]]
  # If an integer was passed, a single-column dataframe would be returned.

  # Finally, use this sample to collect data for starting:
  # Use .values attribute to obtain only the values stored in the Pandas Series
  lagging_current_reactive_power = df_sample['Lagging_Current_Reactive.Power_kVarh_mean'].values
  # df_sample is a pd.DataFrame, whereas df_sample['Lagging_Current_Reactive.Power_kVarh_mean']
  # is a pd.Series object. So, both store column names and types, as well as row indices.
  leading_current_reactive_power = df_sample['Leading_Current_Reactive_Power_kVarh_mean'].values
  co2_tco2 = df_sample['CO2(tCO2)_mean'].values
  lagging_current_power_factor = df_sample['Lagging_Current_Power_Factor_mean'].values
  load_type = df_sample['Load_Type_mode'].values

  return lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, load_type


def create_timestamp_array(start_date, total_days, total_hours):
  """Create the array of simulation timestamps from the defined inputs."""
  # Since one day has 24 hours, the simulation proceeds through {total_hours} = 
  total_hours = total_days*24 + total_hours

  # Create a list of timestamps for the simulation:
  timestamps = np.array([(start_date + pd.Timedelta(i, 'h')) for i in range(0, (total_hours + 1))])
  # Goes from 0 (no functioning) to the total_hours + 1 - 1 = total_hours.
  # Each step sums a timedelta of i hours

  total_entries = len(timestamps) # total of values that must be saved.

  return timestamps, total_entries


def check_weekstatus(day_of_week):
  """Verify if the day is weekday or weekend"""
  if day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    return 'Weekday'
  
  elif day_of_week in ['Saturday', 'Sunday']:
    return 'Weekend'


def create_dayofweek_weekstatus(timestamps):
  """Create arrays with day name (day of week) and week status (if it is weekday or weekend).
  They are needed for obtaining a dataframe with the same format as the original data
  used for training the model. The dataset returned for the user will have this format,
  so it can be directly concatenated to the raw dataset. Also, the pipeline used for
  the simulation becomes the same used for treating data and training the model.

  pd.Timestamp.dayofweek attribute returns an integer number representing the day of the week.
  https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html
  pd.Timestamp.day_name() method returns the name of the weekday, such as 'Wednesday'.
  https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.day_name.html#pandas.Timestamp.day_name
  """
  
  # Get array of day names, called day_of_week:
  day_of_week = np.array([time.day_name() for time in timestamps])
  # Create a list mapping the check_weekstatus function to each day_of_week,
  # i.e., apply the function to each element of the array. Then, convert to array:
  weekstatus = np.array(list(map(check_weekstatus, day_of_week)))

  return day_of_week, weekstatus


def calculate_nsm(start_date, timestamps):
  """Calculate the NSM to get a dataset with same format as the original one.
  
  NSM: Number of Seconds from midnight: continuous variable measured in s

  pd.Timestamp.date() method returns a datetime object with same year, month and day.
  Applying again the function pd.Timestamp to this object returns a Timestamp at 00:00:00.
  https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.date.html#pandas.Timestamp.date

  So, the difference between a timestamp and this last one will be a timedelta in relation
  to midnight. In seconds, the value is the NSM.

  pd.Timedelta.total_seconds() method returns the timedelta as equivalent number of total seconds:
  https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.total_seconds.html#pandas.Timedelta.total_seconds
  """
  
  # Get midnight timestamp correspondent to start_date:
  midnight = pd.Timestamp(pd.Timestamp(start_date).date())

  # get a list of timedeltas, subtracting each timestamp from midnight:
  timedeltas = [(timestamp - midnight) for timestamp in timestamps]
  # apply total_seconds method to each element and convert to np.ndarray.
  # It could be a single step, but the syntax would be harder to read:
  nsm = np.array([timedelta.total_seconds() for timedelta in timedeltas])

  return nsm


def convert_input_vars_to_arrays(total_entries, lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, load_type):
  """Create arrays for each of the other input variables, corresponding to each simulation timestamp.
  This function converts each one of the inputs to an array
  """

  # for each one of the timestamps, let's create a 1-dimensional array completely constant, containing
  # these values.
  # For that, we can use numpy full, which creates a constant matrix
  # https://numpy.org/doc/stable/reference/generated/numpy.full.html
  lagging_current_reactive_power = np.full((total_entries,), lagging_current_reactive_power)
  leading_current_reactive_power = np.full((total_entries,), leading_current_reactive_power)
  co2_tco2 = np.full((total_entries,), co2_tco2)
  lagging_current_power_factor = np.full((total_entries,), lagging_current_power_factor)
  load_type = np.full((total_entries,), load_type)

  return lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, load_type


def obtain_simulation_df(timestamps, lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, leading_current_power_factor, nsm, weekstatus, day_of_week, load_type):
  """Get a Pandas dataframe from each one of the arrays."""
  
  sim_df = pd.DataFrame(data = {

    "timestamp": timestamps,
    "lagging_current_reactive_power_kvarh": lagging_current_reactive_power,
    "leading_current_reactive_power_kvarh": leading_current_reactive_power,
    "co2_tco2": co2_tco2,
    "lagging_current_power_factor": lagging_current_power_factor,
    "leading_current_power_factor": leading_current_power_factor,
    "nsm": nsm,
    "weekstatus": weekstatus,
    "day_of_week": day_of_week,
    "load_type": load_type

  })

  return sim_df


def add_variation_to_features(dataset, possible_ranges):
  """ Add the random noise to each continous input to add a source
  of variation.
  - Check if all inputs are within the valid ranges. If a value is outside
  the boundaries, correct it. Notice that the boundaries force the simulated
  data to get closer to the distorted distributions: if a boundary is close to
  the mean value, then there will be more data on the other side of the distribution,
  introducing skewness and kurtosis.
  - Finally, apply the linear correlation to calculate the
  leading current power factor, and add the variation to this feature.
  """
  
  checked_variables = ['lagging_current_reactive_power_kvarh', 
                      'leading_current_reactive_power_kvarh',
                      'co2_tco2', 'lagging_current_power_factor']
  
  for var in checked_variables:
    # Check if the variables are within the valid range. If they are not, raise an error:
    var_array = np.array(dataset[var])
    var_min = possible_ranges[var]['min']
    var_max = possible_ranges[var]['max']
    std = possible_ranges[var]['std']
    
    # Add a random noise to this feature:
    var_array = random_noise(var_array, std)
    # Check if array contains a value above the max or below the minimum.
    var_array = correct_vals_out_of_bounds(var_array, var_min, var_max)

    # Add the corrected array to the dataset:
    dataset[var] = var_array
  
  # Now that the simulation passed through the checking phase, apply the linear correlation:
  leading_current_reactive_power = np.array(dataset['leading_current_reactive_power'])
  leading_current_power_factor = calculate_leading_current_power_factor(leading_current_reactive_power, possible_ranges)
  # Add to the dataset:
  dataset['leading_current_power_factor'] = leading_current_power_factor

  return dataset
