import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb # Required to load XGBoost model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration and Model/Scaler Loading ---
st.set_page_config(layout="wide", page_title="Moqhaka Overdue Bill Prediction")

# Define paths for your saved model and scaler
MODEL_PATH = 'model_artifacts/best_xgboost_model.pkl'
SCALER_PATH = 'model_artifacts/scaler.pkl'

@st.cache_resource # Caches the model and scaler to load only once
def load_model_and_scaler():
    """Loads the trained XGBoost model and StandardScaler from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Required model or scaler files not found. Please ensure 'model_artifacts' directory and its contents ({MODEL_PATH}, {SCALER_PATH}) are in the same directory as this app.py file.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()

# --- Title and Introduction ---
st.title("💡 Moqhaka Local Municipality: Proactive Overdue Bill Prediction Prototype")
st.write("""
This interactive prototype dashboard helps Moqhaka Local Municipality identify bills that are most likely to become overdue,
allowing for proactive intervention and improved revenue collection.
""")
st.markdown("---")

# --- Dashboard Controls (Sidebar) ---
st.sidebar.header("Dashboard Controls")
prediction_threshold = st.sidebar.slider(
    "Select Prediction Threshold for 'Overdue'",
    min_value=0.01, max_value=0.99, value=0.50, step=0.01,
    help="Adjust this threshold to see how it affects the model's performance (Precision vs. Recall). A lower threshold increases Recall but may decrease Precision."
)
st.sidebar.markdown("---")
st.sidebar.info("This is a prototype. Data used is illustrative. For full functionality, actual municipal data would be loaded.")


# --- Actual Data Loading and Prediction ---
@st.cache_data # Caches data loading/processing to avoid re-running on every interaction
def load_and_predict_data(scaler_obj, model_obj):
    """
    Loads actual data from a CSV, scales numerical features, and generates predictions.
    """
    DATA_FILE_PATH = 'data/df_engineered_features.parquet' # Customize this path if your file is elsewhere

    # --- COMPLETE LIST OF 108 FEATURE COLUMNS ---
    # This list is taken directly from your provided feature list.
    FEATURE_COLUMNS = [
        'initial_balance_fwd', 'bill_amount_due', 'is_estimated_reading',
        'has_payment_agreement', 'num_dependents', 'is_prepaid_electricity',
        'overdue_likelihood', 'balance_to_due_ratio', 'days_to_due_date',
        'prev_bill_amount_due', 'prev_amount_paid', 'prev_payment_ratio',
        'prev_num_days_overdue_at_snapshot', 'prev_is_overdue_target',
        'change_bill_amount_due', 'num_service_types_on_bill',
        'num_payments_last_30D', 'rolling_sum_amount_paid_30D',
        'rolling_mean_amount_paid_30D', 'rolling_std_amount_paid_30D',
        'rolling_sum_bill_amount_due_30D', 'rolling_mean_bill_amount_due_30D',
        'rolling_std_bill_amount_due_30D',
        'rolling_sum_num_days_overdue_at_snapshot_30D',
        'rolling_mean_num_days_overdue_at_snapshot_30D',
        'rolling_std_num_days_overdue_at_snapshot_30D',
        'rolling_sum_payment_ratio_30D', 'rolling_mean_payment_ratio_30D',
        'rolling_std_payment_ratio_30D',
        'rolling_sum_initial_balance_fwd_30D',
        'rolling_mean_initial_balance_fwd_30D',
        'rolling_std_initial_balance_fwd_30D',
        'num_estimated_bills_last_30D', 'num_payments_last_90D',
        'rolling_sum_amount_paid_90D', 'rolling_mean_amount_paid_90D',
        'rolling_std_amount_paid_90D', 'rolling_sum_bill_amount_due_90D',
        'rolling_mean_bill_amount_due_90D', 'rolling_std_bill_amount_due_90D',
        'rolling_sum_num_days_overdue_at_snapshot_90D',
        'rolling_mean_num_days_overdue_at_snapshot_90D',
        'rolling_std_num_days_overdue_at_snapshot_90D',
        'rolling_sum_payment_ratio_90D', 'rolling_mean_payment_ratio_90D',
        'rolling_std_payment_ratio_90D',
        'rolling_sum_initial_balance_fwd_90D',
        'rolling_mean_initial_balance_fwd_90D',
        'rolling_std_initial_balance_fwd_90D',
        'num_estimated_bills_last_90D', 'num_payments_last_180D',
        'rolling_sum_amount_paid_180D', 'rolling_mean_amount_paid_180D',
        'rolling_std_amount_paid_180D', 'rolling_sum_bill_amount_due_180D',
        'rolling_mean_bill_amount_due_180D',
        'rolling_std_bill_amount_due_180D',
        'rolling_sum_num_days_overdue_at_snapshot_180D',
        'rolling_mean_num_days_overdue_at_snapshot_180D',
        'rolling_std_num_days_overdue_at_snapshot_180D',
        'rolling_sum_payment_ratio_180D',
        'rolling_mean_payment_ratio_180D',
        'rolling_std_payment_ratio_180D',
        'rolling_sum_initial_balance_fwd_180D',
        'rolling_mean_initial_balance_fwd_180D',
        'rolling_std_initial_balance_fwd_180D',
        'num_estimated_bills_last_180D', 'num_payments_last_365D',
        'rolling_sum_amount_paid_365D', 'rolling_mean_amount_paid_365D',
        'rolling_std_amount_paid_365D', 'rolling_sum_bill_amount_due_365D',
        'rolling_mean_bill_amount_due_365D',
        'rolling_std_bill_amount_due_365D',
        'rolling_sum_num_days_overdue_at_snapshot_365D',
        'rolling_mean_num_days_overdue_at_snapshot_365D',
        'rolling_std_num_days_overdue_at_snapshot_365D',
        'rolling_sum_payment_ratio_365D', 'rolling_mean_payment_ratio_365D',
        'rolling_std_payment_ratio_365D',
        'rolling_sum_initial_balance_fwd_365D',
        'rolling_mean_initial_balance_fwd_365D',
        'rolling_std_initial_balance_fwd_365D',
        'num_estimated_bills_last_365D', 'prev_initial_balance_fwd_1m',
        'change_initial_balance_1m', 'balance_to_avg_bill_ratio',
        'customer_tenure_days', 'customer_type_Commercial',
        'customer_type_Government', 'customer_type_Indigent',
        'customer_type_Residential', 'property_value_category_High',
        'property_value_category_Low', 'property_value_category_Medium',
        'ward_Ward_1', 'ward_Ward_10', 'ward_Ward_2', 'ward_Ward_3',
        'ward_Ward_4', 'ward_Ward_5', 'ward_Ward_6', 'ward_Ward_7',
        'ward_Ward_8', 'ward_Ward_9',
        'historical_payment_behavior_Average',
        'historical_payment_behavior_Good',
        'historical_payment_behavior_Poor'
    ]

    TARGET_COLUMN = 'is_overdue_target' # Confirmed from your previous output

    # --- NUMERICAL COLUMNS FOR SCALING ---
    # This list is inferred based on common scaling practices (excluding binary/one-hot).
    # Double-check this against your notebook's exact StandardScaler usage.
    NUMERICAL_COLS_FOR_SCALING = [
        'initial_balance_fwd', 'bill_amount_due', 'num_dependents',
        'overdue_likelihood', 'balance_to_due_ratio', 'days_to_due_date',
        'prev_bill_amount_due', 'prev_amount_paid', 'prev_payment_ratio',
        'prev_num_days_overdue_at_snapshot', 'change_bill_amount_due',
        'num_service_types_on_bill', 'num_payments_last_30D',
        'rolling_sum_amount_paid_30D', 'rolling_mean_amount_paid_30D',
        'rolling_std_amount_paid_30D', 'rolling_sum_bill_amount_due_30D',
        'rolling_mean_bill_amount_due_30D', 'rolling_std_bill_amount_due_30D',
        'rolling_sum_num_days_overdue_at_snapshot_30D',
        'rolling_mean_num_days_overdue_at_snapshot_30D',
        'rolling_std_num_days_overdue_at_snapshot_30D',
        'rolling_sum_payment_ratio_30D', 'rolling_mean_payment_ratio_30D',
        'rolling_std_payment_ratio_30D',
        'rolling_sum_initial_balance_fwd_30D',
        'rolling_mean_initial_balance_fwd_30D',
        'rolling_std_initial_balance_fwd_30D',
        'num_estimated_bills_last_30D', 'num_payments_last_90D',
        'rolling_sum_amount_paid_90D', 'rolling_mean_amount_paid_90D',
        'rolling_std_amount_paid_90D', 'rolling_sum_bill_amount_due_90D',
        'rolling_mean_bill_amount_due_90D', 'rolling_std_bill_amount_due_90D',
        'rolling_sum_num_days_overdue_at_snapshot_90D',
        'rolling_mean_num_days_overdue_at_snapshot_90D',
        'rolling_std_num_days_overdue_at_snapshot_90D',
        'rolling_sum_payment_ratio_90D', 'rolling_mean_payment_ratio_90D',
        'rolling_std_payment_ratio_90D',
        'rolling_sum_initial_balance_fwd_90D',
        'rolling_mean_initial_balance_fwd_90D',
        'rolling_std_initial_balance_fwd_90D',
        'num_estimated_bills_last_90D', 'num_payments_last_180D',
        'rolling_sum_amount_paid_180D', 'rolling_mean_amount_paid_180D',
        'rolling_std_amount_paid_180D', 'rolling_sum_bill_amount_due_180D',
        'rolling_mean_bill_amount_due_180D',
        'rolling_std_bill_amount_due_180D',
        'rolling_sum_num_days_overdue_at_snapshot_180D',
        'rolling_mean_num_days_overdue_at_snapshot_180D',
        'rolling_std_num_days_overdue_at_snapshot_180D',
        'rolling_sum_payment_ratio_180D',
        'rolling_mean_payment_ratio_180D',
        'rolling_std_payment_ratio_180D',
        'rolling_sum_initial_balance_fwd_180D',
        'rolling_mean_initial_balance_fwd_180D',
        'rolling_std_initial_balance_fwd_180D',
        'num_estimated_bills_last_180D', 'num_payments_last_365D',
        'rolling_sum_amount_paid_365D', 'rolling_mean_amount_paid_365D',
        'rolling_std_amount_paid_365D', 'rolling_sum_bill_amount_due_365D',
        'rolling_mean_bill_amount_due_365D',
        'rolling_std_bill_amount_due_365D',
        'rolling_sum_num_days_overdue_at_snapshot_365D',
        'rolling_mean_num_days_overdue_at_snapshot_365D',
        'rolling_std_num_days_overdue_at_snapshot_365D',
        'rolling_sum_payment_ratio_365D', 'rolling_mean_payment_ratio_365D',
        'rolling_std_payment_ratio_365D',
        'rolling_sum_initial_balance_fwd_365D',
        'rolling_mean_initial_balance_fwd_365D',
        'rolling_std_initial_balance_fwd_365D',
        'num_estimated_bills_last_365D', 'prev_initial_balance_fwd_1m',
        'change_initial_balance_1m', 'balance_to_avg_bill_ratio',
        'customer_tenure_days'
    ]


    # --- 2. Load the data ---
    try:
        # Assuming your test data includes customer_id, bill_id, and your target column.
        # If 'bill_id' or 'customer_id' are not in your actual data, adjust the
        # `display_cols` variable in the "Actionable List" section below.
        df = pd.read_csv(DATA_FILE_PATH)
        st.success(f"Successfully loaded data from {DATA_FILE_PATH} with {len(df)} records for prediction.")
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_FILE_PATH}. Please ensure '{DATA_FILE_PATH}' is in your GitHub repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # --- 3. Prepare features (X) and actual labels (y_true) for prediction and evaluation ---
    data_for_prediction_df = df.copy()

    try:
        X_data = data_for_prediction_df[FEATURE_COLUMNS]
        y_true = data_for_prediction_df[TARGET_COLUMN] # Actual labels for evaluation
    except KeyError as e:
        st.error(f"Missing expected column in data file: {e}. Please double-check FEATURE_COLUMNS and TARGET_COLUMN lists in app.py against your '{DATA_FILE_PATH}'.")
        st.stop()

    # Scale numerical features using the loaded scaler
    X_data_scaled = X_data.copy()
    try:
        X_data_scaled[NUMERICAL_COLS_FOR_SCALING] = scaler_obj.transform(X_data[NUMERICAL_COLS_FOR_SCALING])
    except Exception as e:
        st.error(f"Error scaling data. Ensure NUMERICAL_COLS_FOR_SCALING are correct and match the features and type (numerical) in your loaded data: {e}")
        st.stop()


    # --- 4. Get probabilities from the loaded model ---
    probabilities = model_obj.predict_proba(X_data_scaled)[:, 1]

    # --- 5. Add predictions and actuals back to the DataFrame for display ---
    data_for_prediction_df['predicted_probability_overdue'] = probabilities
    data_for_prediction_df['actual_overdue_status'] = y_true

    return data_for_prediction_df, y_true

# Load and predict using actual data
data_for_prediction, y_true_for_evaluation = load_and_predict_data(scaler, model)

# --- Make predictions based on the selected threshold ---
y_pred_threshold = (data_for_prediction['predicted_probability_overdue'] >= prediction_threshold).astype(int)


# --- Section 1: Model Performance Summary ---
st.header("📊 Model Performance Summary (at current threshold)")

col1, col2, col3, col4 = st.columns(4)

accuracy = accuracy_score(y_true_for_evaluation, y_pred_threshold)
precision = precision_score(y_true_for_evaluation, y_pred_threshold, zero_division=0)
recall = recall_score(y_true_for_evaluation, y_pred_threshold, zero_division=0)
f1 = f1_score(y_true_for_evaluation, y_pred_threshold, zero_division=0)
roc_auc = roc_auc_score(y_true_for_evaluation, data_for_prediction['predicted_probability_overdue'])


col1.metric("Recall (Overdue Detected)", f"{recall:.2%}", help="Percentage of actual overdue bills correctly identified by the model.")
col2.metric("Precision (Accuracy of Flags)", f"{precision:.2%}", help="Percentage of flagged bills that are truly overdue.")
col3.metric("F1-Score", f"{f1:.2f}", help="Harmonic mean of Precision and Recall. Higher is better.")
col4.metric("Accuracy", f"{accuracy:.2%}", help="Overall correctness of predictions.")

st.markdown("---")

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true_for_evaluation, y_pred_threshold)

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Not Overdue (0)', 'Predicted Overdue (1)'],
            yticklabels=['Actual Not Overdue (0)', 'Actual Overdue (1)'], ax=ax_cm)
ax_cm.set_title(f'Confusion Matrix (Threshold: {prediction_threshold:.2f})')
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('True Label')
st.pyplot(fig_cm)

st.markdown("---")


# --- Section 2: Actionable List of Bills Predicted as Overdue ---
st.header("📝 Actionable List: Bills Predicted as Overdue")
st.write(f"Displaying bills with a predicted probability of being overdue >= {prediction_threshold:.2f}.")

overdue_predictions_df = data_for_prediction[y_pred_threshold == 1].copy()
overdue_predictions_df = overdue_predictions_df.sort_values(by='predicted_probability_overdue', ascending=False)

if not overdue_predictions_df.empty:
    # Include 'customer_id', 'bill_id', 'predicted_probability_overdue', 'actual_overdue_status'
    # and the full set of FEATURE_COLUMNS for comprehensive display.
    # Ensure 'bill_id' and 'customer_id' are in your actual data if you want them displayed.
    display_cols = ['customer_id', 'bill_id', 'predicted_probability_overdue', 'actual_overdue_status']
    # Add features to display, excluding ones already in display_cols
    display_cols.extend([col for col in FEATURE_COLUMNS if col not in display_cols])

    # Filter `overdue_predictions_df` to only include columns that actually exist in it,
    # to prevent errors if some display_cols are missing from the dataframe itself.
    existing_display_cols = [col for col in display_cols if col in overdue_predictions_df.columns]

    st.dataframe(
        overdue_predictions_df[existing_display_cols].head(100) # Displays top 100. Consider using st.data_editor or pagination for very large lists.
    )
    st.write(f"Total bills flagged as overdue: {len(overdue_predictions_df)}")
    st.info("Staff can use this list to prioritize proactive outreach.")
else:
    st.info("No bills predicted as overdue at the current threshold. Try lowering the threshold if you expect some.")


st.markdown("---")

# --- Section 3: Model Insights - Feature Importance ---
st.header("💡 Model Insights: Top Contributing Factors")
st.write("Understanding which factors most influence the model's predictions.")

# Get feature importance and ensure feature names are aligned
if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
    feature_names_for_importance = model.feature_names_in_
elif hasattr(model, 'feature_names') and model.feature_names is not None:
     feature_names_for_importance = model.feature_names
else:
    # Fallback: Use the FEATURE_COLUMNS defined for data loading in the app
    feature_names_for_importance = FEATURE_COLUMNS

feature_importances = model.feature_importances_

if feature_names_for_importance is not None and len(feature_names_for_importance) == len(feature_importances):
    importance_df = pd.DataFrame({'Feature': feature_names_for_importance, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax_fi, palette='viridis')
    ax_fi.set_title('Top 10 Feature Importances')
    ax_fi.set_xlabel('Importance Score')
    ax_fi.set_ylabel('Feature')
    st.pyplot(fig_fi)
else:
    st.warning("Could not display feature importances. Ensure feature names are correctly retrieved from the model or provided.")

st.markdown("---")

# --- Section 4: Probability Distribution ---
st.header("📈 Predicted Probability Distribution")
st.write("Distribution of predicted probabilities across all bills.")

fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
sns.histplot(data_for_prediction['predicted_probability_overdue'], bins=50, kde=True, ax=ax_prob)
ax_prob.axvline(prediction_threshold, color='red', linestyle='--', label=f'Threshold: {prediction_threshold:.2f}')
ax_prob.set_title('Distribution of Predicted Probabilities')
ax_prob.set_xlabel('Predicted Probability of Being Overdue')
ax_prob.set_ylabel('Number of Bills')
ax_prob.legend()
st.pyplot(fig_prob)

st.markdown("---")
st.success("Prototype dashboard ready!")

# --- END OF APP.PY ---