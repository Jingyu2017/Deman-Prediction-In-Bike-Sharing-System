# Deman-Prediction-In-Bike-Sharing-System

#### input file 
- "bike_2_years" folder contains Santander bike journey data in csv format from 25 May 2016 to 22 May 2018. One file contains observations worth of one week.

- "bike_aws" folder contains real-time station status  of 786  Santander Cycle docking stations 24/7 from 2018-06-23 to 2018-08-16 via Transport for Londons unified API. 
- "DockingLocation.csv" contains detailed information of each station.
- "holiday" folder contains list of bank holidays in UK.
- "weather" contains London's hourly meteorological data.

#### script file

p.s. Figure x refers to figure in the thesis.



- "utc_to_local_timezone2.py" --- in order to keep consistency among data from different sources, UTC time has to be converted to local time in weather dataset. 
- "TimeOfDay_CheckOutNumber.py" --- source code for generating Figure 7 and Figure 8
- "Temp_WindSpeed_CheckOutNumber.py" --- source code for generating Figure 10 and Figure 11
- "Weather_CheckOutNumber.py" --- source code for generating Figure 9
- "Patterns_among_clusters.py" --- part 1 is the source code for generating Figure 12; the function of part 2 is to find out all the docking stations ever used at least once; the function of part 3 is to convert british national system to wgs84, and merge two data set related to station coordinations into one complete reference; part 4 and part 5 generate input attributes for clustering algorithm; part 6 is the implemention of KMeans clustering (see Section 4.4 in the thesis); part 7 and 8 are concerned with visualisation (Figure 15 and Figure 20)
- "cluster_measurement.py" ---source code for generating Figure 14, the Silhouette score and Calinski-Harabaz score for clustering results.
- "model_input.py" ---source code for generating a preprocessed input matrix.
- "Predict.py" --- Part I is code that yieilds prediction results using Sklearn packages; Part II is to generate feature importances with forests of trees (See Figure 21)(refenrence: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
- "evaluation.py" --- source code for generating R squared score for predictions of each cluster (See Figure 20)​	