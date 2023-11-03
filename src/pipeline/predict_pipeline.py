import sys
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)
    

#no_of_dependents	education	self_employed	income_annum
# loan_amount	loan_term	cibil_score	residential_assets_value	commercial_assets_value	luxury_assets_value	bank_asset_value
class CustomData:
    def __init__(self,
                 #loan_id:int,
                 no_of_dependents:int,
                 self_employed:str,
                 education:str,
                 income_annum:int,
                 loan_amount:int,
                 loan_term:int,
                 cibil_score:int,
                 residential_assets_value:int,
                 commercial_assets_value:int,
                 luxury_assets_value:int,
                 bank_asset_value:int):
        #self.loan_id = loan_id
        self.no_of_dependent = no_of_dependents
        self.education = education
        self.self_employed = self_employed
        self.income_annum = income_annum
        self.loan_amount = loan_amount
        self.loan_term = loan_term
        self.cibil_score = cibil_score
        self.residential_assets_value = residential_assets_value
        self.commercial_assets_value = commercial_assets_value
        self.luxury_assets_value = luxury_assets_value
        self.bank_asset_value = bank_asset_value

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                #" loan_id":[self.loan_id],
                " no_of_dependents":[self.no_of_dependent],
                " education":[self.education],
                " self_employed":[self.self_employed],
                " income_annum":[self.income_annum],
                " loan_amount":[self.loan_amount],
                " loan_term":[self.loan_term],
                " cibil_score":[self.cibil_score],
                " residential_assets_value":[self.residential_assets_value],
                " commercial_assets_value":[self.commercial_assets_value],
                " luxury_assets_value":[self.luxury_assets_value],
                " bank_asset_value":[self.bank_asset_value],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)

