from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from .idsw.datafetch.core import InvalidInputsError
from .idsw.datafetch.pipes import upload_to_or_download_file_from_colab, export_pd_dataframe_as_excel
from .idsw.etl.characterize import time_series_vis

from .models import load_models

from .transformvariables import simulation_pipeline
from .utils import (load_df_and_ranges,
                    random_start,
                    create_timestamp_array,
                    create_dayofweek_weekstatus,
                    calculate_nsm,
                    convert_input_vars_to_arrays,
                    obtain_simulation_df,
                    add_variation_to_features
                    )


@dataclass
class GlobalVars:
  """
    Store Global variables to use on simulation for the model.
    The variables are stored on a higher context so that they are not lost
  """
  
  # Counter of simulations:
  simulation_counter = 0
  # Start a list of exported tables:
  exported_tables = []

  # load models and store them
  kmeans_model, encoder_decoder_tf_model = load_models()

  # Load original dataframe and allowed ranges for the variables:
  df, possible_ranges = load_df_and_ranges()

  # Start variables with random values:
  lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, load_type = random_start(df)
  
  """DEFAULT PARAMETERS - USER MAY RUN SIMULATION WITHOUT UPDATING DATA"""
  # default value of start date will be the instant:
  start_date = pd.Timestamp(datetime.now())
  total_days = 1
  total_hours = 0

  # Obtain arrays related to the timestamps:
  timestamps, total_entries = create_timestamp_array(start_date, total_days, total_hours)
  day_of_week, weekstatus = create_dayofweek_weekstatus(timestamps)
  nsm = calculate_nsm(start_date, timestamps)

  # Convert the input variables to arrays (one value for each timestamp)
  lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, load_type = convert_input_vars_to_arrays(total_entries, lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, load_type)
  
  # Now, create a dataframe for the simulations:
  sim_df = obtain_simulation_df(timestamps, lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, leading_current_power_factor, nsm, weekstatus, day_of_week, load_type)
  # Finally, add variation to this dataframe:
  sim_df = add_variation_to_features(sim_df, possible_ranges)


def update_with_inputs(var1, var2, var3, var4, var5, var6, var7, var8):
  """Update the GlobalVars with the user inputs
  var1: start_date: day for starting simulation (selected on date picker).
  var2: total_days: int with total days of simulation (manual input)
  var3: total_hours: int with total hours of simulation, beyond total days.
    - Manual input
    User wants the factory to run for {total_days} + {total_hours}
  
  var4: lagging_current_reactive_power (slider)
  var5: leading_current_reactive_power (slider)
  var6: co2_tco2 (slider)
  var7: lagging_current_power_factor (slider)

  var8: load_type (str): selected on the dropdown.
  """

  # Run only if one of the inputs is different from the stored in memory.
  start_date = pd.Timestamp(var1)
  total_days = int(var2)
  total_hours = int(var3)

  # Several global variables are arrays with constant values. Since we cannot compare the full
  # array with a value, due to ambiguity, let's compare only its 1st value.
  boolean_check = ((GlobalVars.start_date != start_date) | (GlobalVars.total_days != total_days) | 
      (GlobalVars.total_hours != total_hours) | (GlobalVars.lagging_current_reactive_power[0] != var4) | 
      (GlobalVars.leading_current_reactive_power[0] != var5) | (GlobalVars.co2_tco2[0] != var6) | 
      (GlobalVars.lagging_current_power_factor[0] != var7) | (GlobalVars.load_type[0] != var8))

  if (boolean_check):
    # Update values on GlobalVars:
    GlobalVars.start_date = start_date
    GlobalVars.total_days = total_days
    GlobalVars.total_hours = total_hours

    # Obtain arrays related to the timestamps:
    timestamps, total_entries = create_timestamp_array(start_date, total_days, total_hours)
    day_of_week, weekstatus = create_dayofweek_weekstatus(timestamps)
    nsm = calculate_nsm(start_date, timestamps)
    # Update values on GlobalVars:
    GlobalVars.timestamps = timestamps
    GlobalVars.total_entries = total_entries
    GlobalVars.day_of_week = day_of_week
    GlobalVars.weekstatus = weekstatus
    GlobalVars.nsm = nsm
    
    # Convert the input variables to arrays (one value for each timestamp)
    lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, load_type = convert_input_vars_to_arrays(total_entries, var4, var5, var6, var7, var8)
    # Update values on GlobalVars:
    GlobalVars.lagging_current_reactive_power = lagging_current_reactive_power
    GlobalVars.leading_current_reactive_power = leading_current_reactive_power
    GlobalVars.co2_tco2 = co2_tco2
    GlobalVars.lagging_current_power_factor = lagging_current_power_factor
    GlobalVars.load_type = load_type

    # Now, create a dataframe for the simulations:
    sim_df = obtain_simulation_df(timestamps, lagging_current_reactive_power, leading_current_reactive_power, co2_tco2, lagging_current_power_factor, leading_current_power_factor, nsm, weekstatus, day_of_week, load_type)
    # Finally, add variation to this dataframe:
    sim_df = add_variation_to_features(sim_df, GlobalVars.possible_ranges)
    # Update values on GlobalVars:
    GlobalVars.sim_df = sim_df

    # This function updates GlobalVars with user inputs and returns the sim_df ready for the simulations.
    # Update the counter and return dataframe:
    GlobalVars.simulation_counter = GlobalVars.simulation_counter + 1
    return sim_df

  else:
    # return the dataframe already in the memory:
    # Update the counter and return dataframe:
    GlobalVars.simulation_counter = GlobalVars.simulation_counter + 1
    return GlobalVars.sim_df
    

def run_simulation(var1, var2, var3, var4, var5, var6, var7, var8):
  """Run all the pipelines to obtain a full simulation.
  At the end, store in a list of dictionaries in GlobalVars, that will be used for exporting a 
  consolidated Excel file with all simulations.
  : params var1, var2, var3, var4, var5, var6, var7, var8: user defined parameters.
  """

  # Get initial dataframe with user defined inputs:
  sim_df = update_with_inputs(var1, var2, var3, var4, var5, var6, var7, var8)
  # Run simulation pipeline:
  sim_df = simulation_pipeline(sim_df, GlobalVars.possible_ranges, GlobalVars.kmeans_model, GlobalVars.encoder_decoder_tf_model)
  # Update on GlobalVars:
  GlobalVars.sim_df = sim_df


  # Get the simulation counting:
  simulation_counter = GlobalVars.simulation_counter
  # Get start date:
  start_date = GlobalVars.start_date
  # Obtain sheet name:
  sheet_name = "sim" + str(simulation_counter) + "_" + str(start_date)
  
  # Get a dictionary for exporting the table:
  table_dict = {'dataframe_obj_to_be_exported': sim_df, 
                    'excel_sheet_name': sheet_name}

  # Append the dictionary on the list of exported tables:
  GlobalVars.exported_tables = GlobalVars.exported_tables.append(table_dict)


  completion_msg = f"""
    
    
    ----------------------------------------------------------------------
                      STEEL INDUSTRY DIGITAL TWIN TERMINAL



    # SIMULATION COMPLETED!


    START = {GlobalVars.start_date}
    SIMULATION #{simulation_counter} 
    SIMULATION FOR {GlobalVars.total_days} DAYS AND {GlobalVars.total_hours}




    ------------------------------------------------------------------------

    
    
    
    """
  
  print(completion_msg)
  try:
        # only works in Jupyter Notebook:
        from IPython.display import display
        display(sim_df)
            
  except: # regular mode
        print(sim_df)


def visualize_usage_kwh(export_images = True):
  """Plot the Usage kWh for the simulations
  : param: export_images = True keep True to
  export the image files and download them.
  """

  # Loop through each simulation:
  for table_dict in GlobalVars.exported_tables:
    msg = f"""
    
    
      ----------------------------------------------------------------------
                        STEEL INDUSTRY DIGITAL TWIN TERMINAL



                            ENERGY CONSUME (kWh)


      SIMULATION DATA IN {table_dict['excel_sheet_name']}
      
      ------------------------------------------------------------------------

      """

    print(msg)

    df = table_dict['dataframe_obj_to_be_exported']
    timestamp = df['timestamp']
    usage_kwh = df['usage_kwh']

    DATA_IN_SAME_COLUMN = False
    DATASET = None
    COLUMN_WITH_PREDICT_VAR_X = 'X'
    COLUMN_WITH_RESPONSE_VAR_Y = 'Y'
    COLUMN_WITH_LABELS = 'label_column'
    LIST_OF_DICTIONARIES_WITH_SERIES_TO_ANALYZE = [
        
        {'x': timestamp, 'y': usage_kwh, 'lab': 'usage_kwh'}, 
    ]
    X_AXIS_ROTATION = 70
    Y_AXIS_ROTATION = 0
    GRID = True
    ADD_SPLINE_LINES = True
    ADD_SCATTER_DOTS = False
    HORIZONTAL_AXIS_TITLE = 'Timestamp'
    VERTICAL_AXIS_TITLE = 'kWh'
    PLOT_TITLE = table_dict['excel_sheet_name']

    EXPORT_PNG = export_images
    DIRECTORY_TO_SAVE = ""
    FILE_NAME = table_dict['excel_sheet_name']
    PNG_RESOLUTION_DPI = 330
    
    time_series_vis (data_in_same_column = DATA_IN_SAME_COLUMN, df = DATASET, column_with_predict_var_x = COLUMN_WITH_PREDICT_VAR_X, column_with_response_var_y = COLUMN_WITH_RESPONSE_VAR_Y, column_with_labels = COLUMN_WITH_LABELS, list_of_dictionaries_with_series_to_analyze = LIST_OF_DICTIONARIES_WITH_SERIES_TO_ANALYZE, x_axis_rotation = X_AXIS_ROTATION, y_axis_rotation = Y_AXIS_ROTATION, grid = GRID, add_splines_lines = ADD_SPLINE_LINES, add_scatter_dots = ADD_SCATTER_DOTS, horizontal_axis_title = HORIZONTAL_AXIS_TITLE, vertical_axis_title = VERTICAL_AXIS_TITLE, plot_title = PLOT_TITLE, export_png = EXPORT_PNG, directory_to_save = DIRECTORY_TO_SAVE, file_name = FILE_NAME, png_resolution_dpi = PNG_RESOLUTION_DPI)

    if (export_images):
      # Download the png file saved in Colab environment:
      ACTION = 'download'
      FILE_TO_DOWNLOAD_FROM_COLAB = (table_dict['excel_sheet_name'] + ".png")
      upload_to_or_download_file_from_colab (action = ACTION, file_to_download_from_colab = FILE_TO_DOWNLOAD_FROM_COLAB)


def download_excel_with_data():
  """Download Excel file containing all the tables generated from simulations."""

  # Create Excel file and store it in Colab's memory:
  FILE_NAME_WITHOUT_EXTENSION = "datasets"
  EXPORTED_TABLES = GlobalVars.exported_tables
  FILE_DIRECTORY_PATH = ""
  export_pd_dataframe_as_excel (file_name_without_extension = FILE_NAME_WITHOUT_EXTENSION, exported_tables = EXPORTED_TABLES, file_directory_path = FILE_DIRECTORY_PATH)

  # Download the file:
  ACTION = 'download'
  FILE_TO_DOWNLOAD_FROM_COLAB = "datasets.xlsx"
  upload_to_or_download_file_from_colab (action = ACTION, file_to_download_from_colab = FILE_TO_DOWNLOAD_FROM_COLAB)
