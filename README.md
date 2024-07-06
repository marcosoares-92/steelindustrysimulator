# STEEL INDUSTRY DIGITAL TWIN
## Simulate a steel industry energy consume using Deep Learning.

#### The digital twin is designed to reproduce the behavior from a DAEWOO Steel Co. Ltd facility in Gwangyang, South Korea, which made its data public.


`Acess user interface in:`

	https://colab.research.google.com/github/marcosoares-92/steelindustrysimulator/blob/main/steelindustry_digitaltwin.ipynb


- `Attention:` this simulator will only run when in a directory called `steelindustrysimulator`.

`Contact:`
- Marco Cesar Prado Soares, Data Scientist Specialist @ Bayer Crop Science LATAM
	- marcosoares.feq@gmail.com
	- marco.soares@bayer.com

## This simulator applies advanced AI (deep learning) technologies to reproduce the operation of a small-scale steel industry.

- This factory produces several types of coils, steel plates, and iron plates. 
- The information on electricity consumption is held in a cloud-based system. 
- The information on energy consumption of the industry is stored on the website of the Korea Electric Power Corporation (pccs.kepco.go.kr); and the perspectives on daily, monthly, and annual data are calculated and shown.

All this information was used for creating the algorithms that will reproduce the energy consume behavior based on your user inputs.

# YOUR GOAL HERE IS TO MINIMIZE THE ENERGY CONSUMPTION, WHICH IS SHOWN IN kWh.

## For that, you will define:
- The day of starting the plant simulation (which can be today).
- The total days and hours for running the plant in the defined conditions
	- Default is 1 day and 0 hours, i.e., 24h of operation.
        
- Plant operation parameters:
	- Lagging Current reactive power, in kVArh; 
        - Leading Current reactive power, in kVArh; 
        - tCO2(CO2), in ppm; 
        - Lagging Current power factor, in %;
        - Load Type: Light Load, Medium Load, Maximum Load.

## And will obtain the Response variable:
- Energy consumption, in kWh

## The simulator returns also the following information:
- Leading Current Power factor, in %; 
- Number of Seconds from midnight (NSM), in seconds (s); 
- Week status: if the simulated day is 'Weekend' or 'Weekday'; 
- Day of week: 'Sunday', 'Monday', ..., 'Saturday'. 


# Industrial Data Science Workflow
## Industrial Data Science Workflow: full workflow for ETL, statistics, and Machine learning modelling of (usually) time-stamped industrial facilities data.
### Not only applicable to monitoring quality and industrial facilities systems, the package can be applied to data manipulation, characterization and modelling of different numeric and categorical datasets to boost your work and replace tradicional tools like SAS, Minitab and Statistica software.

Check the project Github: https://github.com/marcosoares-92/IndustrialDataScienceWorkflow

Authors:
- Marco Cesar Prado Soares, Data Scientist Specialist at Bayer (Crop Science)
  - marcosoares.feq@gmail.com

- Gabriel Fernandes Luz, Senior Data Scientist
  - gfluz94@gmail.com


# Steel Industry Energy Consumption Dataset
- Check in UCI Datasets [https://archive.ics.uci.edu/ml/datasets/Steel+Industry+Energy+Consumption+Dataset]

## Abstract: 
- The data is collected from a smart small-scale steel industry in South Korea. 

## Data Set Information:
- The information gathered is from the DAEWOO Steel Co. Ltd in Gwangyang, South Korea. 
- It produces several types of coils, steel plates, and iron plates. 
- The information on electricity consumption is held in a cloud-based system. 
- The information on energy consumption of the industry is stored on the website of the Korea Electric Power Corporation (pccs.kepco.go.kr); 
	- and the perspectives on daily, monthly, and annual data are calculated and shown.

## Attribute Information: 
### Data Variables:
 
	- Type Measurement; 
	- Industry Energy Consumption Continuous kWh; 
	- Lagging Current reactive power Continuous kVarh; 
	- Leading Current reactive power Continuous kVarh; 
	- tCO2(CO2) Continuous ppm; 
	- Lagging Current power factor Continuous %; 
	- Leading Current Power factor Continuous %; 
	- Number of Seconds from midnight Continuous S; 
	- Week status Categorical (Weekend (0) or a Weekday(1)); 
	- Day of week Categorical Sunday, Monday, ..., Saturday; 
	- Load Type Categorical Light Load, Medium Load, Maximum Load.

##### Shape = (35040, 11)
##### Response: energy consumption

### Source:
 - Sathishkumar V E,
 - Department of Information and Communication Engineering, Sunchon National University, Suncheon.
 - Republic of Korea.
 - Email: srisathishkumarve@gmail.com
