from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

#home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            no_of_dependents = int(request.form.get("no_of_dependents")),
            self_employed = " "+request.form.get("self_employed"),
            education = " "+request.form.get("education"),
            income_annum = request.form.get("income_annum"),
            loan_amount = request.form.get("loan_amount"),
            loan_term = request.form.get("loan_term"),
            cibil_score = request.form.get("cibil_score"),
            residential_assets_value = request.form.get("residential_assets_value"),
            commercial_assets_value = request.form.get("commercial_assets_value"),
            luxury_assets_value = request.form.get("luxury_assets_value"),
            bank_asset_value = request.form.get("bank_asset_value"),
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0" ,debug=True)
