from flask import Flask, request, jsonify
import joblib
import numpy as np
import sys

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('best_gb_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract and calculate the required features
        loan_amount_requested = data.get("loan_amount_requested")
        existing_debt_or_other_payments = data.get("existing_debt_or_other_payments")
        annual_income = data.get("annual_income")
        monthly_income = data.get("monthly_income")
        total_outstanding_debt = data.get("total_outstanding_debt")
        recoveries = data.get("recoveries")
        total_rec_int = data.get("total_rec_int")
        total_current_balance = data.get("total_current_balance")
        total_credit_limit = data.get("total_credit_limit")
        batch_enrolled = data.get("batch_enrolled")
        emp_length = data.get("emp_length")

        # Perform backend calculations
        total_payment = loan_amount_requested + existing_debt_or_other_payments
        income_to_loan_ratio = annual_income / loan_amount_requested
        dti_revol_util = total_outstanding_debt / monthly_income
        total_recovery = recoveries + total_rec_int
        balance_to_credit_ratio = total_current_balance / total_credit_limit
        recoveries_to_balance_ratio = recoveries / (total_current_balance + 1)
        batch_enrolled_to_total_rec_int = batch_enrolled / (total_rec_int + 1)
        loan_amnt_total_rec_int_ratio = loan_amount_requested / (total_rec_int + 1)
        emp_length_missing = int(emp_length is None)

        # Create feature array
        features = np.array([[
            total_payment,
            income_to_loan_ratio,
            dti_revol_util,
            total_recovery,
            balance_to_credit_ratio,
            recoveries_to_balance_ratio,
            batch_enrolled_to_total_rec_int,
            loan_amnt_total_rec_int_ratio,
            emp_length_missing
        ]])

        # Make prediction
        prediction = model.predict(features)
        
        return jsonify({
            'prediction': int(prediction[0])  # Return as int for JSON compatibility
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
