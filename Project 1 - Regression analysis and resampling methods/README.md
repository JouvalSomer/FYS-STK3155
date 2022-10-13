
# README for project 1:
This project is for regression methods in Project 01 in FYS-STK4155 at UiO.
October 2022.

Authors: Jouval Somer and Stig Patey


### Regression analysis and resampling methods
Included are four main python files for the project
- func.py
- plot.py
- frank_main.py
- terrain_main.py

For internal testing purposes we created:
- predictors.py
- scaledemo.py


### Output files from terminal:
- franke__results
- terrain_results

Note that output from multiple runs are not provided.


In addition is the Report it self, available on the associated folder.

### func.py:
This file includes all the functions used. That includes the OLS, RIDGE and LASSO function which then include bootstrap and cross validation.

### plot.py
This file includes all the plot function used.

### frank_main.py and terrain_main.py
The files to execute the regresson for Franke and Terrain.


### predictors.py
Just to calculate the number of variables in the design matrix given a n-th order polynom and plots.


### scaledemo.py
Testing of scaling, own code for validation.

## Note:
> We observed some anomalities when running multiple LASSO plots at once. Mainly the LASSO plot copied some values from the previous runs. This did not occur for Ridge. The problem was solved by commenting out plots and run only one LASSO plot at the time.
