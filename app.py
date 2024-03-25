from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route("/")
def main():

    data = request.get_json()

    df = pd.DataFrame([data], index=[0])

    prediction = model.predict_proba(df[model.feature_names_in_])[:, 0] * 1000

    return jsonify({'score': int(prediction), 'cod_cli': data['COD_CLI']})

if __name__ == '__main__':
    app.run(debug=True)
