### Import functions ###
import inspect
import numpy as np
import pandas as pd

### Functions ###
from utils.Selector import *
from utils.Prediction import *

### Function ###
def LearningProcedure(SimulationConfigInputUpdated):
    """
    Executes an iterative active learning or greedy sampling loop.

    Args:
        SimulationConfigInputUpdated (dict): A dictionary containing the configuration and state for the learning loop.

    Returns:
        dict: A dictionary containing the results of the learning procedure with the following keys:
            - ErrorVec (dict): A dictionary where keys are metric names ('RMSE', 'MAE', 'R2', 'CC') and 
            values are lists of the metric's value at each iteration of the loop.
            - SelectedObservationHistory (list): A list of the indices of observations selected from the candidate pool
              in the order they were chosen.
    """

    ### Set Up ###
    i = 0
    ErrorVecs = {
    'Test_Set': {'RMSE': [], 'MAE': [], 'R2': [], 'CC': []},
    'Full_Pool':    {'RMSE': [], 'MAE': [], 'R2': [], 'CC': []}
    }
    SelectedObservationHistory = []

    ### Initialize Model ###
    ModelClass = globals().get(SimulationConfigInputUpdated["ModelType"], None)
    model_init_args = {k: v for k, v in SimulationConfigInputUpdated.items()
                       if k in inspect.signature(ModelClass.__init__).parameters}
    predictor_model = ModelClass(**model_init_args)
    SimulationConfigInputUpdated['Model'] = predictor_model

    ### Initialize Selector ###
    if 'df_Candidate' in SimulationConfigInputUpdated:
        SimulationConfigInputUpdated['initial_candidate_size'] = len(SimulationConfigInputUpdated['df_Candidate'])
    SelectorClass = globals().get(SimulationConfigInputUpdated["SelectorType"], None)
    selector_init_args = {k: v for k, v in SimulationConfigInputUpdated.items()
                          if k in inspect.signature(SelectorClass.__init__).parameters}
    selector_model = SelectorClass(**selector_init_args)

    ### Algorithm ###
    while len(SimulationConfigInputUpdated["df_Candidate"]) > 0:

        ## Get features and target for the current training set ##
        X_train_df, y_train_series = get_features_and_target(
            df=SimulationConfigInputUpdated["df_Train"],
            target_column_name="Y"
            )

        ## Prediction Model ##
        predictor_model.fit(X_train_df=X_train_df, y_train_series=y_train_series)
        
        ## Calculate Test Error (both from the paper and the standard way)

        # 1. Hold-Out Test Error
        TestSetErrorOutput = HoldOutErrorFunction(InputModel=predictor_model,
                                                df_Test=SimulationConfigInputUpdated["df_Test"])
        for metric_name, value in TestSetErrorOutput.items():
            ErrorVecs['Test_Set'][metric_name].append(value)

        # 2. Full_Pool Error
        FullPoolErrorOuputs = FullPoolErrorFunction(InputModel=predictor_model,
                                                df_Train=SimulationConfigInputUpdated["df_Train"],
                                                df_Candidate=SimulationConfigInputUpdated["df_Candidate"])
        for metric_name, value in FullPoolErrorOuputs.items():
            ErrorVecs['Full_Pool'][metric_name].append(value)

        ## Sampling Procedure ##
        SelectorFuncOutput = selector_model.select(df_Candidate=SimulationConfigInputUpdated["df_Candidate"],
                                                df_Train=SimulationConfigInputUpdated["df_Train"],
                                                Model=predictor_model,
                                                current_rmse=FullPoolErrorOuputs["RMSE"])

        ## Query selected observation ##
        QueryObservationIndex = SelectorFuncOutput["IndexRecommendation"]
        QueryObservation = SimulationConfigInputUpdated["df_Candidate"].loc[QueryObservationIndex]
        SelectedObservationHistory.append(QueryObservationIndex)

        ## Update Train and Candidate Sets ##
        SimulationConfigInputUpdated["df_Train"] = pd.concat([SimulationConfigInputUpdated["df_Train"], QueryObservation])
        SimulationConfigInputUpdated["df_Candidate"] = SimulationConfigInputUpdated["df_Candidate"].drop(QueryObservationIndex)
        
        ## Increase iteration ##
        i+=1

    ### Output ###
    LearningProcedureOutput = {"ErrorVecs": ErrorVecs,
                               "SelectedObservationHistory": SelectedObservationHistory}
                              
    return LearningProcedureOutput