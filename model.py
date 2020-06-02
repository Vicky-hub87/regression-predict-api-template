"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Pickup Lat','Pickup Long',
                                        #'Destination Lat','Destination Long']]
    # ------------------------------------------------------------------------
    Train = pd.read_csv('data/train_data.csv')
    Riders = pd.read_csv('data/riders.csv')

    def percent_missing(Train):
        output = dict()
        for column in Train.columns:
             output[column] = np.round(Train[column].isnull().sum() / len(Train[column]) * 100, 2)
        return output

    Train = Train.drop(columns = ["Precipitation in millimeters"])
    Train.fillna(Train.mean(),inplace=True)

    Y = Train[['Arrival at Destination - Day of Month','Arrival at Destination - Weekday (Mo = 1)','Arrival at Destination - Time',
           'Time from Pickup to Arrival']]
    Train = Train.drop(['Arrival at Destination - Day of Month','Arrival at Destination - Weekday (Mo = 1)','Arrival at Destination - Time',
           'Time from Pickup to Arrival'],axis = 1)
    Combine_Train_Test = pd.concat([Train, feature_vector_df])
    Combine_Train_Test = Combine_Train_Test.drop(['Vehicle Type','Platform Type','Personal or Business','User Id','Rider Id','Order No'],axis = 1)

    #convert time to datetime
    Combine_Train_Test['Placement - Time'] = Combine_Train_Test['Placement - Time'].astype('datetime64')
    Combine_Train_Test['Confirmation - Time'] = Combine_Train_Test['Confirmation - Time'].astype('datetime64')
    Combine_Train_Test['Arrival at Pickup - Time'] = Combine_Train_Test['Arrival at Pickup - Time'].astype('datetime64')
    Combine_Train_Test['Pickup - Time'] = Combine_Train_Test['Pickup - Time'].astype('datetime64')

    #convert hours and min from time
    Combine_Train_Test['Placement_Hour'] = Combine_Train_Test['Placement - Time'].dt.hour
    Combine_Train_Test['Placement_Minute'] = Combine_Train_Test['Placement - Time'].dt.minute
    Combine_Train_Test['Confirmation_Hour'] = Combine_Train_Test['Confirmation - Time'].dt.hour
    Combine_Train_Test['Confirmation_Minute'] = Combine_Train_Test['Confirmation - Time'].dt.minute
    Combine_Train_Test['Arrival_Hour'] = Combine_Train_Test['Arrival at Pickup - Time'].dt.hour
    Combine_Train_Test['Arrival_Minute'] = Combine_Train_Test['Arrival at Pickup - Time'].dt.minute
    Combine_Train_Test['Pickup_Hour'] = Combine_Train_Test['Pickup - Time'].dt.hour
    Combine_Train_Test['Pickup_Minute'] = Combine_Train_Test['Pickup - Time'].dt.minute

    #addtions and subtractions of time
    Combine_Train_Test['process_time_hour'] = Combine_Train_Test['Confirmation - Time'].dt.hour - Combine_Train_Test['Placement - Time'].dt.hour
    Combine_Train_Test['process_time_minute'] = Combine_Train_Test['Confirmation - Time'].dt.minute - Combine_Train_Test['Placement - Time'].dt.minute

    Combine_Train_Test['cust_wait1_time_hour'] = Combine_Train_Test['Arrival at Pickup - Time'].dt.hour - Combine_Train_Test['Placement - Time'].dt.hour
    Combine_Train_Test['cust_wait1_time_minute'] = Combine_Train_Test['Arrival at Pickup - Time'].dt.minute - Combine_Train_Test['Placement - Time'].dt.minute

    Combine_Train_Test['cust_wait2_time_hour'] = Combine_Train_Test['Arrival at Pickup - Time'].dt.hour - Combine_Train_Test['Confirmation - Time'].dt.hour
    Combine_Train_Test['cust_wait2_time_minute'] = Combine_Train_Test['Arrival at Pickup - Time'].dt.minute - Combine_Train_Test['Confirmation - Time'].dt.minute

    Combine_Train_Test['standby_hoours'] = Combine_Train_Test['Pickup - Time'].dt.hour - Combine_Train_Test['Arrival at Pickup - Time'].dt.hour
    Combine_Train_Test['standby_minutes'] = Combine_Train_Test['Pickup - Time'].dt.minute - Combine_Train_Test['Arrival at Pickup - Time'].dt.minute

    Combine_Train_Test['cust_wait3_time_hour'] = Combine_Train_Test['Pickup - Time'].dt.hour - Combine_Train_Test['Confirmation - Time'].dt.hour
    Combine_Train_Test['cust_wait3_time_minute'] = Combine_Train_Test['Pickup - Time'].dt.minute - Combine_Train_Test['Confirmation - Time'].dt.minute

    Combine_Train_Test['cust_wait4_time_hour'] = Combine_Train_Test['Pickup - Time'].dt.hour - Combine_Train_Test['Placement - Time'].dt.hour
    Combine_Train_Test['cust_wait4_time_minute'] = Combine_Train_Test['Pickup - Time'].dt.minute - Combine_Train_Test['Placement - Time'].dt.minute

    Combine_Train_Test = Combine_Train_Test.drop(['Placement - Time','Confirmation - Time','Arrival at Pickup - Time','Pickup - Time'],axis = 1)
     

    def get_part_hr(hour):
        if (hour > 4) and (hour <= 8):
            return 'Early Morning'
        elif (hour > 8) and (hour < 12 ):
            return 'Morning'
        elif (hour >= 12) and (hour <= 16):
            return'Noon'
        elif (hour > 16) and (hour <= 20):
            return 'Evening'
        elif (hour > 20) and (hour <= 22):
            return'Night'
        else:
            return'Late Night'
    
    Combine_Train_Test['Placement_Time_Day'] = Combine_Train_Test['Placement_Hour'].apply(get_part_hr)
    Combine_Train_Test['Confirmation_Time_Day'] = Combine_Train_Test['Confirmation_Hour'].apply(get_part_hr)
    Combine_Train_Test['Arrival_Time_Day'] = Combine_Train_Test['Arrival_Hour'].apply(get_part_hr)
    Combine_Train_Test['Pickup_Time_Day'] = Combine_Train_Test['Pickup_Hour'].apply(get_part_hr)

     #get weekend of weekday
    Combine_Train_Test['Actual Day of Placement'] = np.where(Combine_Train_Test['Placement - Weekday (Mo = 1)'] < 6,'Weekday','Weekend')
    Combine_Train_Test['Actual Day of Confirmation'] = np.where(Combine_Train_Test['Confirmation - Weekday (Mo = 1)'] < 6,'Weekday','Weekend')
    Combine_Train_Test['Actual Day of Arrival'] = np.where(Combine_Train_Test['Arrival at Pickup - Weekday (Mo = 1)'] < 6,'Weekday','Weekend')
    Combine_Train_Test['Actual Day of Pickup'] = np.where(Combine_Train_Test['Pickup - Weekday (Mo = 1)'] < 6,'Weekday','Weekend')

    #get time of month
    Place_Day_Month = []
    # Using a for loop to populate the list
    for Each_Day in Combine_Train_Test['Placement - Day of Month']:
        if (Each_Day > 0 and Each_Day <=7):
            Place_Day_Month.append('1st Week')        
        elif (Each_Day > 7 and Each_Day <=15):
             Place_Day_Month.append('2nd Week')
        elif (Each_Day > 15 and Each_Day <=23):
             Place_Day_Month.append('3rd Week')
        else:
            Place_Day_Month.append('4th Week')
        
    Combine_Train_Test['Placement_Day into Weeks'] = Place_Day_Month


    Confirm_Day_Month = []
    # Using a for loop to populate the list
    for Each_Day in Combine_Train_Test['Confirmation - Day of Month']:
        if (Each_Day > 0 and Each_Day <=7):
            Confirm_Day_Month.append('1st Week')        
        elif (Each_Day > 7 and Each_Day <=15):
             Confirm_Day_Month.append('2nd Week')
        elif (Each_Day > 15 and Each_Day <=23):
             Confirm_Day_Month.append('3rd Week')
        else:
            Confirm_Day_Month.append('4th Week')

    Combine_Train_Test['Confirmation_Day into Weeks'] = Confirm_Day_Month

    Arrival_Pickup_Day_Month = []
    # Using a for loop to populate the list
    for Each_Day in Combine_Train_Test['Arrival at Pickup - Day of Month']:
        if (Each_Day > 0 and Each_Day <=7):
             Arrival_Pickup_Day_Month.append('1st Week')        
        elif (Each_Day > 7 and Each_Day <=15):
            Arrival_Pickup_Day_Month.append('2nd Week')
        elif (Each_Day > 15 and Each_Day <=23):
             Arrival_Pickup_Day_Month.append('3rd Week')
        else:
             Arrival_Pickup_Day_Month.append('4th Week')
        
    Combine_Train_Test['Arrival_Pickup_Day into Weeks'] = Arrival_Pickup_Day_Month

    Pickup_Day_Month = []
    # Using a for loop to populate the list
    for Each_Day in Combine_Train_Test['Pickup - Day of Month']:
        if (Each_Day > 0 and Each_Day <=7):
             Pickup_Day_Month.append('1st Week')        
        elif (Each_Day > 7 and Each_Day <=15):
             Pickup_Day_Month.append('2nd Week')
        elif (Each_Day > 15 and Each_Day <=23):
             Pickup_Day_Month.append('3rd Week')
        else:
             Pickup_Day_Month.append('4th Week')
        
    Combine_Train_Test['Pickup_Day into Weeks'] = Pickup_Day_Month


    #Creating New Feature = Grouping Temperature into Average High, Average Low and Normal
    Temperature_Deg = []

    # Using a for loop to populate the list
    for Degree in Combine_Train_Test['Temperature']:
        if (Degree >= 15) and (Degree <= 27):
             Temperature_Deg.append('Average High Temperature')
        elif (Degree >= 12) and (Degree <= 22 ):
             Temperature_Deg.append('Average Low Temperature')
        else:
             Temperature_Deg.append('Normal Temperature')
    
    Combine_Train_Test['Temperature Condition'] = Temperature_Deg

    Combine_Train_Test = pd.get_dummies(Combine_Train_Test)


    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
