import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb  # Required to load XGBoost model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

# --- Configuration and Model/Scaler Loading ---
st.set_page_config(layout="wide", page_title="Moqhaka Overdue Bill Prediction")

# Define paths for your saved model and scaler
MODEL_PATH = "model_artifacts/best_xgboost_model.pkl"
SCALER_PATH = "model_artifacts/scaler.pkl"
DATA_FOLDER_NAME = "data"  # Define the folder name once
DATA_FILE_NAME = "df_engineered_features.parquet"  # Define the data file name

# --- IMPORTANT: Define FEATURE_COLUMNS and SCALED_NUMERICAL_COLS globally ---
# These lists must be accessible by all parts of the script, including display sections.
# They MUST precisely match the features used to train the model, in the correct order.

TARGET_COLUMN = "is_overdue_target"

# --- MODIFIED: Re-added 'num_payments_last_X_D' and 'num_estimated_bills_last_X_D' ---
# Also adjusted SCALED_NUMERICAL_COLS based on your training script output
FEATURE_COLUMNS = [
    "initial_balance_fwd",
    "bill_amount_due",
    "is_estimated_reading",
    "has_payment_agreement",
    "num_dependents",
    "is_prepaid_electricity",
    # "overdue_likelihood", # Confirmed NOT in numerical_cols for training, so leave removed
    "balance_to_due_ratio",
    "days_to_due_date",
    "prev_bill_amount_due",
    "prev_amount_paid",
    "prev_payment_ratio",
    "prev_num_days_overdue_at_snapshot",
    "prev_is_overdue_target",
    "change_bill_amount_due",
    "num_service_types_on_bill",
    "num_payments_last_30D",  # RE-ADDED based on your training data
    "rolling_sum_amount_paid_30D",
    "rolling_mean_amount_paid_30D",
    "rolling_std_amount_paid_30D",
    "rolling_sum_bill_amount_due_30D",
    "rolling_mean_bill_amount_due_30D",
    "rolling_std_bill_amount_due_30D",
    "rolling_sum_num_days_overdue_at_snapshot_30D",
    "rolling_mean_num_days_overdue_at_snapshot_30D",
    "rolling_std_num_days_overdue_at_snapshot_30D",
    "rolling_sum_payment_ratio_30D",
    "rolling_mean_payment_ratio_30D",
    "rolling_std_payment_ratio_30D",
    "rolling_sum_initial_balance_fwd_30D",
    "rolling_mean_initial_balance_fwd_30D",
    "rolling_std_initial_balance_fwd_30D",
    "num_estimated_bills_last_30D",  # RE-ADDED based on your training data
    "num_payments_last_90D",  # RE-ADDED based on your training data
    "rolling_sum_amount_paid_90D",
    "rolling_mean_amount_paid_90D",
    "rolling_std_amount_paid_90D",
    "rolling_sum_bill_amount_due_90D",
    "rolling_mean_bill_amount_due_90D",
    "rolling_std_bill_amount_due_90D",
    "rolling_sum_num_days_overdue_at_snapshot_90D",
    "rolling_mean_num_days_overdue_at_snapshot_90D",
    "rolling_std_num_days_overdue_at_snapshot_90D",
    "rolling_sum_payment_ratio_90D",
    "rolling_mean_payment_ratio_90D",
    "rolling_std_payment_ratio_90D",
    "rolling_sum_initial_balance_fwd_90D",
    "rolling_mean_initial_balance_fwd_90D",
    "rolling_std_initial_balance_fwd_90D",
    "num_estimated_bills_last_90D",  # RE-ADDED based on your training data
    "num_payments_last_180D",  # RE-ADDED based on your training data
    "rolling_sum_amount_paid_180D",
    "rolling_mean_amount_paid_180D",
    "rolling_std_amount_paid_180D",
    "rolling_sum_bill_amount_due_180D",
    "rolling_mean_bill_amount_due_180D",
    "rolling_std_bill_amount_due_180D",
    "rolling_sum_num_days_overdue_at_snapshot_180D",
    "rolling_mean_num_days_overdue_at_snapshot_180D",
    "rolling_std_num_days_overdue_at_snapshot_180D",
    "rolling_sum_payment_ratio_180D",
    "rolling_mean_payment_ratio_180D",
    "rolling_std_payment_ratio_180D",
    "rolling_sum_initial_balance_fwd_180D",
    "rolling_mean_initial_balance_fwd_180D",
    "rolling_std_initial_balance_fwd_180D",
    "num_estimated_bills_last_180D",  # RE-ADDED based on your training data
    "num_payments_last_365D",  # RE-ADDED based on your training data
    "rolling_sum_amount_paid_365D",
    "rolling_mean_amount_paid_365D",
    "rolling_std_amount_paid_365D",
    "rolling_sum_bill_amount_due_365D",
    "rolling_mean_bill_amount_due_365D",
    "rolling_std_bill_amount_due_365D",
    "rolling_sum_num_days_overdue_at_snapshot_365D",
    "rolling_mean_num_days_overdue_at_snapshot_365D",
    "rolling_std_num_days_overdue_at_snapshot_365D",
    "rolling_sum_payment_ratio_365D",
    "rolling_mean_payment_ratio_365D",
    "rolling_std_payment_ratio_365D",
    "rolling_sum_initial_balance_fwd_365D",
    "rolling_mean_initial_balance_fwd_365D",
    "rolling_std_initial_balance_fwd_365D",
    "num_estimated_bills_last_365D",  # RE-ADDED based on your training data
    "prev_initial_balance_fwd_1m",
    "change_initial_balance_1m",
    "balance_to_avg_bill_ratio",
    "customer_tenure_days",
    "customer_type_Commercial",
    "customer_type_Government",
    "customer_type_Indigent",
    "customer_type_Residential",
    "property_value_category_High",
    "property_value_category_Low",
    "property_value_category_Medium",
    "ward_Ward_1",
    "ward_Ward_10",
    "ward_Ward_2",
    "ward_Ward_3",
    "ward_Ward_4",
    "ward_Ward_5",
    "ward_Ward_6",
    "ward_Ward_7",
    "ward_Ward_8",
    "ward_Ward_9",
    "historical_payment_behavior_Average",
    "historical_payment_behavior_Good",
    "historical_payment_behavior_Poor",
]

# --- MODIFIED: Adjusted SCALED_NUMERICAL_COLS based on your training script output ---
# 'num_payments_last_30D' and 'num_estimated_bills_last_30D' are binary, NOT scaled.
# 'num_payments_last_90D', 'num_estimated_bills_last_90D', etc. ARE scaled.
SCALED_NUMERICAL_COLS = [
    "initial_balance_fwd",
    "bill_amount_due",
    "num_dependents",
    # "overdue_likelihood", # Was confirmed as not in numerical_cols
    "balance_to_due_ratio",
    "days_to_due_date",
    "prev_bill_amount_due",
    "prev_amount_paid",
    "prev_payment_ratio",
    "prev_num_days_overdue_at_snapshot",
    "change_bill_amount_due",
    "num_service_types_on_bill",
    # 'num_payments_last_30D' (binary, not scaled)
    "rolling_sum_amount_paid_30D",
    "rolling_mean_amount_paid_30D",
    "rolling_std_amount_paid_30D",
    "rolling_sum_bill_amount_due_30D",
    "rolling_mean_bill_amount_due_30D",
    "rolling_std_bill_amount_due_30D",
    "rolling_sum_num_days_overdue_at_snapshot_30D",
    "rolling_mean_num_days_overdue_at_snapshot_30D",
    "rolling_std_num_days_overdue_at_snapshot_30D",
    "rolling_sum_payment_ratio_30D",
    "rolling_mean_payment_ratio_30D",
    "rolling_std_payment_ratio_30D",
    "rolling_sum_initial_balance_fwd_30D",
    "rolling_mean_initial_balance_fwd_30D",
    "rolling_std_initial_balance_fwd_30D",
    # 'num_estimated_bills_last_30D' (binary, not scaled)
    "num_payments_last_90D",  # Scaled
    "rolling_sum_amount_paid_90D",
    "rolling_mean_amount_paid_90D",
    "rolling_std_amount_paid_90D",
    "rolling_sum_bill_amount_due_90D",
    "rolling_mean_bill_amount_due_90D",
    "rolling_std_bill_amount_due_90D",
    "rolling_sum_num_days_overdue_at_snapshot_90D",
    "rolling_mean_num_days_overdue_at_snapshot_90D",
    "rolling_std_num_days_overdue_at_snapshot_90D",
    "rolling_sum_payment_ratio_90D",
    "rolling_mean_payment_ratio_90D",
    "rolling_std_payment_ratio_90D",
    "rolling_sum_initial_balance_fwd_90D",
    "rolling_mean_initial_balance_fwd_90D",
    "rolling_std_initial_balance_fwd_90D",
    "num_estimated_bills_last_90D",  # Scaled
    "num_payments_last_180D",  # Scaled
    "rolling_sum_amount_paid_180D",
    "rolling_mean_amount_paid_180D",
    "rolling_std_amount_paid_180D",
    "rolling_sum_bill_amount_due_180D",
    "rolling_mean_bill_amount_due_180D",
    "rolling_std_bill_amount_due_180D",
    "rolling_sum_num_days_overdue_at_snapshot_180D",
    "rolling_mean_num_days_overdue_at_snapshot_180D",
    "rolling_std_num_days_overdue_at_snapshot_180D",
    "rolling_sum_payment_ratio_180D",
    "rolling_mean_payment_ratio_180D",
    "rolling_std_payment_ratio_180D",
    "rolling_sum_initial_balance_fwd_180D",
    "rolling_mean_initial_balance_fwd_180D",
    "rolling_std_initial_balance_fwd_180D",
    "num_estimated_bills_last_180D",  # Scaled
    "num_payments_last_365D",  # Scaled
    "rolling_sum_amount_paid_365D",
    "rolling_mean_amount_paid_365D",
    "rolling_std_amount_paid_365D",
    "rolling_sum_bill_amount_due_365D",
    "rolling_mean_bill_amount_due_365D",
    "rolling_std_bill_amount_due_365D",
    "rolling_sum_num_days_overdue_at_snapshot_365D",
    "rolling_mean_num_days_overdue_at_snapshot_365D",
    "rolling_std_num_days_overdue_at_snapshot_365D",
    "rolling_sum_payment_ratio_365D",
    "rolling_mean_payment_ratio_365D",
    "rolling_std_payment_ratio_365D",
    "rolling_sum_initial_balance_fwd_365D",
    "rolling_mean_initial_balance_fwd_365D",
    "rolling_std_initial_balance_fwd_365D",
    "num_estimated_bills_last_365D",  # Scaled
    "prev_initial_balance_fwd_1m",
    "change_initial_balance_1m",
    "balance_to_avg_bill_ratio",
    "customer_tenure_days",
]
# --- End of global definitions ---


@st.cache_resource  # Caches the model and scaler to load only once
def load_model_and_scaler():
    """Loads the trained XGBoost model and StandardScaler from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(
            f"DEBUG: Model and Scaler loaded successfully from {MODEL_PATH} and {SCALER_PATH}."
        )
        return model, scaler
    except FileNotFoundError:
        st.error(
            f"Required model or scaler files not found. Please ensure 'model_artifacts' directory and its contents ({MODEL_PATH}, {SCALER_PATH}) are in the same directory as this app.py file."
        )
        print(
            f"ERROR: Model/Scaler FileNotFoundError: {MODEL_PATH} or {SCALER_PATH} not found."
        )
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        print(f"ERROR: Detailed error loading model or scaler: {e}")
        st.stop()


model, scaler = load_model_and_scaler()

# --- Title and Introduction ---
st.title("💡 Moqhaka Local Municipality: Proactive Overdue Bill Prediction Prototype")
st.write(
    """
This interactive prototype dashboard helps Moqhaka Local Municipality identify bills that are most likely to become overdue,
allowing for proactive intervention and improved revenue collection.
"""
)
st.markdown("---")

# --- Dashboard Controls (Sidebar) ---
st.sidebar.header("Dashboard Controls")
prediction_threshold = st.sidebar.slider(
    "Select Prediction Threshold for 'Overdue'",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Adjust this threshold to see how it affects the model's performance (Precision vs. Recall). A lower threshold increases Recall but may decrease Precision.",
)

# --- Start of Filter Actionable List (Moved up for clarity in logical flow) ---
st.sidebar.header("Filter Actionable List")

# Customer Type Filter
all_customer_types = ["Commercial", "Government", "Indigent", "Residential"]
selected_customer_types = st.sidebar.multiselect(
    "Filter by Customer Type:", options=all_customer_types, default=all_customer_types
)

# Property Value Category Filter
all_property_values = ["High", "Low", "Medium"]
selected_property_values = st.sidebar.multiselect(
    "Filter by Property Value Category:",
    options=all_property_values,
    default=all_property_values,
)

# Ward Filter (assuming wards are 'Ward_1', 'Ward_2', etc. based on your features)
# You'll need to extract the actual unique wards from your data.
# For now, let's assume a generic way to get them.
# This assumes 'ward_Ward_X' are your column names.
ward_cols = [col for col in FEATURE_COLUMNS if col.startswith("ward_Ward_")]
all_wards = [col.replace("ward_", "") for col in ward_cols]
selected_wards = st.sidebar.multiselect(
    "Filter by Ward:", options=all_wards, default=all_wards
)

# Filter for estimated readings (related to transparency)
filter_estimated_reading = st.sidebar.checkbox(
    "Only show bills with Estimated Reading (is_estimated_reading=1)", value=False
)

st.sidebar.markdown("---")  # Visual separator
st.sidebar.info(
    "This is a prototype. Data used is illustrative. For full functionality, actual municipal data would be loaded."
)
# --- End of Filter Actionable List ---


# --- Helper function for CSV download ---
@st.cache_data  # Cache the conversion to avoid re-running on every page load
def convert_df_to_csv(df):
    # This function takes the DataFrame and converts it to a CSV string
    # It's crucial that 'df' here contains all the rows you want to download.
    return df.to_csv(index=False).encode("utf-8")


# Helper function to get top N SHAP contributors for a row
def get_top_shap_contributors(shap_values_row, feature_names, n=3):
    if shap_values_row is None:
        return "N/A"  # Handle cases where SHAP values weren't computed

    # Create a Series of SHAP values indexed by feature names
    shap_series = pd.Series(shap_values_row, index=feature_names)
    # Sort by absolute SHAP value to find most impactful features
    top_features = shap_series.abs().sort_values(ascending=False).head(n).index.tolist()

    # Format as string for display
    contributors = []
    for feat in top_features:
        value = shap_series[feat]
        sign = "+" if value >= 0 else "-"
        # Show sign and magnitude, ensuring feature name matches the original case in FEATURE_COLUMNS
        contributors.append(f"{feat} ({sign}{abs(value):.2f})")
    return ", ".join(contributors)


# --- Actual Data Loading and Prediction Function ---
@st.cache_data  # Caches data loading/processing to avoid re-running on every interaction
def load_and_predict_data(_scaler_obj, _model_obj):
    """
    Loads actual data from a Parquet file, scales numerical features,
    generates predictions, and calculates SHAP values.
    """
    print("--- Inside load_and_predict_data function ---")

    # Construct the full data file path consistently
    DATA_FULL_PATH = os.path.join(DATA_FOLDER_NAME, DATA_FILE_NAME)

    print(f"DEBUG: Current working directory in app.py: {os.getcwd()}")
    print(f"DEBUG: Checking if '{DATA_FOLDER_NAME}' exists...")
    if os.path.exists(DATA_FOLDER_NAME):
        print(f"DEBUG: '{DATA_FOLDER_NAME}' exists. Contents:")
        for item in os.listdir(DATA_FOLDER_NAME):
            item_path = os.path.join(DATA_FOLDER_NAME, item)
            if os.path.isfile(item_path):
                print(
                    f"DEBUG:   File: {item}, Size: {os.path.getsize(item_path)} bytes"
                )
            else:
                print(f"DEBUG:   Directory: {item}")
    else:
        print(f"DEBUG: '{DATA_FOLDER_NAME}' DOES NOT EXIST!")

    # --- 1. Load the data ---
    try:
        df = pd.read_parquet(DATA_FULL_PATH)
        st.success(
            f"Successfully loaded data from {DATA_FULL_PATH} with {len(df)} records for prediction."
        )
        print(f"DEBUG: Parquet file loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        st.error(
            f"Data file not found at {DATA_FULL_PATH}. Please ensure '{DATA_FULL_PATH}' is in your GitHub repository."
        )
        print(f"ERROR: Data FileNotFoundError: {DATA_FULL_PATH} not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        print(f"ERROR: Detailed error during parquet load: {e}")
        st.stop()

    # --- 2. Prepare features (X) and actual labels (y_true) for prediction and evaluation ---
    data_for_prediction_df = df.copy()

    try:
        # Crucially, select columns in the order defined by FEATURE_COLUMNS
        X_data = data_for_prediction_df[FEATURE_COLUMNS]

        if TARGET_COLUMN in data_for_prediction_df.columns:
            y_true = data_for_prediction_df[
                TARGET_COLUMN
            ]  # Actual labels for evaluation
            print(f"DEBUG: Features (X_data) extracted. Shape: {X_data.shape}")
            print(f"DEBUG: Target (y_true) extracted. Shape: {y_true.shape}")
        else:
            st.warning(
                f"Target column '{TARGET_COLUMN}' not found in loaded data. Cannot perform full evaluation metrics."
            )
            print(f"WARNING: Target column '{TARGET_COLUMN}' not found in loaded data.")
            y_true = pd.Series(
                np.zeros(len(df)), index=df.index
            )  # Dummy y_true if not available

    except KeyError as e:
        st.error(
            f"Missing expected column in data file: {e}. Please double-check FEATURE_COLUMNS and TARGET_COLUMN lists in app.py against your '{DATA_FULL_PATH}'."
        )
        print(f"ERROR: KeyError during feature/target extraction: {e}")
        st.stop()

    # Scale numerical features using the loaded scaler
    # Create a copy to avoid SettingWithCopyWarning
    X_data_scaled = X_data.copy()
    try:
        # Only transform the columns specified in SCALED_NUMERICAL_COLS
        # Ensure these columns exist in X_data_scaled before transforming
        cols_to_scale_exist = [
            col for col in SCALED_NUMERICAL_COLS if col in X_data_scaled.columns
        ]
        if len(cols_to_scale_exist) != len(SCALED_NUMERICAL_COLS):
            missing_cols = set(SCALED_NUMERICAL_COLS) - set(cols_to_scale_exist)
            raise ValueError(f"Missing numerical columns for scaling: {missing_cols}")

        X_data_scaled[cols_to_scale_exist] = _scaler_obj.transform(
            X_data_scaled[cols_to_scale_exist]
        )
        print(
            f"DEBUG: Numerical features scaled. X_data_scaled shape after scaling: {X_data_scaled.shape}"
        )
    except Exception as e:
        st.error(
            f"Error scaling data. Ensure SCALED_NUMERICAL_COLS are correct and match the features and type (numerical) in your loaded data: {e}"
        )
        print(f"ERROR: Scaling failed: {e}")
        st.stop()

    # --- 3. Get probabilities from the loaded model ---
    try:
        probabilities = _model_obj.predict_proba(X_data_scaled)[:, 1]
        print(
            f"DEBUG: Model prediction successful. Probabilities shape: {probabilities.shape}"
        )
    except Exception as e:
        st.error(
            f"Error during model prediction: {e}. This might be due to model/data mismatch or environment issues."
        )
        print(f"ERROR: Model prediction failed: {e}")
        st.stop()

    # --- NEW: Calculate SHAP values ---
    print("DEBUG: Calculating SHAP values...")
    try:
        # SHAP Explainer for tree models
        explainer = shap.TreeExplainer(_model_obj)
        # Calculate SHAP values for the scaled data
        shap_values = explainer.shap_values(X_data_scaled)
        print(f"DEBUG: SHAP values calculated. Shape: {shap_values.shape}")
    except Exception as e:
        st.warning(
            f"Could not calculate SHAP values: {e}. Feature explanations will not be available."
        )
        print(f"WARNING: SHAP calculation failed: {e}")
        shap_values = None  # Set to None if calculation fails

    # --- 4. Add predictions and actuals back to the DataFrame for display ---
    data_for_prediction_df["predicted_probability_overdue"] = probabilities
    data_for_prediction_df["actual_overdue_status"] = (
        y_true  # This will add the dummy y_true if it was not found
    )
    print(
        f"DEBUG: Final DataFrame for prediction prepared. Final shape: {data_for_prediction_df.shape}"
    )

    # --- NEW: Also return shap_values and X_data_scaled ---
    return data_for_prediction_df, y_true, shap_values, X_data_scaled


# --- Main app execution flow (calling load_and_predict_data once) ---
print("--- Calling load_and_predict_data outside the function definition ---")
result_of_load_data = load_and_predict_data(scaler, model)

print(f"DEBUG: Type of result_of_load_data: {type(result_of_load_data)}")
# --- FIX: Changed len(result_of_load_data) == 3 to == 4 ---
if isinstance(result_of_load_data, tuple) and len(result_of_load_data) == 4:
    data_for_prediction, y_true_for_evaluation, shap_values_df, X_scaled_for_shap = (
        result_of_load_data  # Unpack shap_values_df and X_scaled_for_shap
    )
    print(
        "DEBUG: Successfully unpacked data_for_prediction, y_true_for_evaluation, shap_values_df, and X_scaled_for_shap."
    )
    # Proceed with the rest of your Streamlit app
else:
    st.error(
        "Application setup failed: Data loading and prediction function did not return expected values. Check logs for details."
    )
    print(
        "ERROR: Application setup failed: load_and_predict_data did not return a 4-item tuple or returned None due to error."
    )
    st.stop()


# --- Make predictions based on the selected threshold ---
y_pred_threshold = (
    data_for_prediction["predicted_probability_overdue"] >= prediction_threshold
).astype(int)


# --- Section 1: Model Performance Summary ---
st.header("📊 Model Performance Summary (at current threshold)")

col1, col2, col3, col4 = st.columns(4)

# Only calculate metrics if y_true_for_evaluation is not a dummy/all zeros due to missing target
if (
    not (y_true_for_evaluation == 0).all() and not (y_true_for_evaluation == 1).all()
):  # Crude check for dummy target
    accuracy = accuracy_score(y_true_for_evaluation, y_pred_threshold)
    precision = precision_score(
        y_true_for_evaluation, y_pred_threshold, zero_division=0
    )
    recall = recall_score(y_true_for_evaluation, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_true_for_evaluation, y_pred_threshold, zero_division=0)
    roc_auc = roc_auc_score(
        y_true_for_evaluation, data_for_prediction["predicted_probability_overdue"]
    )

    col1.metric(
        "Recall (Overdue Detected)",
        f"{recall:.2%}",
        help="Percentage of actual overdue bills correctly identified by the model.",
    )
    col2.metric(
        "Precision (Accuracy of Flags)",
        f"{precision:.2%}",
        help="Percentage of flagged bills that are truly overdue.",
    )
    col3.metric(
        "F1-Score",
        f"{f1:.2f}",
        help="Harmonic mean of Precision and Recall. Higher is better.",
    )
    col4.metric(
        "Accuracy", f"{accuracy:.2%}", help="Overall correctness of predictions."
    )
    st.write(f"ROC AUC: {roc_auc:.4f}")  # Keep this one
else:
    st.warning(
        "Performance metrics (Accuracy, Precision, Recall, F1, ROC AUC) cannot be calculated because the target column ('is_overdue_target') was not found or was empty in the loaded data."
    )
    col1.metric("Recall (Overdue Detected)", "N/A")
    col2.metric("Precision (Accuracy of Flags)", "N/A")
    col3.metric("F1-Score", "N/A")
    col4.metric("Accuracy", "N/A")

# st.write(f"ROC AUC: {roc_auc:.4f}")  # --- FIX: Removed duplicated line ---
st.markdown("---")

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")
if not (y_true_for_evaluation == 0).all() and not (y_true_for_evaluation == 1).all():
    cm = confusion_matrix(y_true_for_evaluation, y_pred_threshold)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Not Overdue (0)", "Predicted Overdue (1)"],
        yticklabels=["Actual Not Overdue (0)", "Actual Overdue (1)"],
        ax=ax_cm,
    )
    ax_cm.set_title(f"Confusion Matrix (Threshold: {prediction_threshold:.2f})")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    st.pyplot(fig_cm)
else:
    st.info("Confusion Matrix not displayed as target column was missing or empty.")

st.markdown("---")


# --- Section 2: Actionable List of Bills Predicted as Overdue ---
st.header("📝 Actionable List: Bills Predicted as Overdue")
st.write(
    f"Displaying bills with a predicted probability of being overdue >= {prediction_threshold:.2f}."
)

# --- FIX: Initialize overdue_predictions_df BEFORE filtering ---
overdue_predictions_initial_df = data_for_prediction[y_pred_threshold == 1].copy()
overdue_predictions_initial_df = overdue_predictions_initial_df.sort_values(
    by="predicted_probability_overdue", ascending=False
)

# Apply filters
filtered_overdue_predictions_df = (
    overdue_predictions_initial_df.copy()
)  # Start with the initially flagged bills

# Apply customer type filter
if selected_customer_types:
    customer_type_filters = [f"customer_type_{ct}" for ct in selected_customer_types]
    # Filter where any of the selected customer type columns are 1
    # Check if the columns actually exist before trying to filter
    existing_type_filters = [
        f for f in customer_type_filters if f in filtered_overdue_predictions_df.columns
    ]
    if existing_type_filters:
        filtered_overdue_predictions_df = filtered_overdue_predictions_df[
            filtered_overdue_predictions_df[existing_type_filters].any(axis=1)
        ]
    else:
        st.warning(
            "Selected customer type filter columns not found in data. Filtering skipped."
        )


# Apply property value category filter
if selected_property_values:
    property_value_filters = [
        f"property_value_category_{pv}" for pv in selected_property_values
    ]
    existing_property_filters = [
        f
        for f in property_value_filters
        if f in filtered_overdue_predictions_df.columns
    ]
    if existing_property_filters:
        filtered_overdue_predictions_df = filtered_overdue_predictions_df[
            filtered_overdue_predictions_df[existing_property_filters].any(axis=1)
        ]
    else:
        st.warning(
            "Selected property value category filter columns not found in data. Filtering skipped."
        )


# Apply ward filter
if selected_wards:
    ward_filters = [f"ward_{w}" for w in selected_wards]
    existing_ward_filters = [
        f for f in ward_filters if f in filtered_overdue_predictions_df.columns
    ]
    if existing_ward_filters:
        filtered_overdue_predictions_df = filtered_overdue_predictions_df[
            filtered_overdue_predictions_df[existing_ward_filters].any(axis=1)
        ]
    else:
        st.warning("Selected ward filter columns not found in data. Filtering skipped.")


# Apply estimated reading filter
if filter_estimated_reading:
    if "is_estimated_reading" in filtered_overdue_predictions_df.columns:
        filtered_overdue_predictions_df = filtered_overdue_predictions_df[
            filtered_overdue_predictions_df["is_estimated_reading"] == 1
        ]
    else:
        st.warning(
            "Column 'is_estimated_reading' not found for filtering. Filtering skipped."
        )


# Now display filtered_overdue_predictions_df
if not filtered_overdue_predictions_df.empty:
    id_cols = ["customer_id", "bill_id"]
    existing_id_cols = [
        col for col in id_cols if col in filtered_overdue_predictions_df.columns
    ]

    display_cols = existing_id_cols + [
        "predicted_probability_overdue",
        "actual_overdue_status",
    ]
    # Add features to display, excluding ones already in display_cols
    display_cols.extend(
        [
            col
            for col in FEATURE_COLUMNS
            if col not in display_cols
            and col in filtered_overdue_predictions_df.columns
        ]
    )

    existing_display_cols = [
        col for col in display_cols if col in filtered_overdue_predictions_df.columns
    ]

    # --- NEW: Add Top Contributing Factors to the DataFrame ---
    if shap_values_df is not None:
        # Map the index of filtered_overdue_predictions_df to the original index of data_for_prediction to get correct SHAP rows
        # This is crucial because filtered_overdue_predictions_df is a subset of data_for_prediction
        # We need the original index to align with the SHAP values array
        original_indices_in_full_df = filtered_overdue_predictions_df.index.tolist()

        # This applies the helper function row-wise to generate the new column
        filtered_overdue_predictions_df["Top_Risk_Factors"] = [
            get_top_shap_contributors(
                shap_values_df[data_for_prediction.index.get_loc(idx)], FEATURE_COLUMNS
            )
            for idx in original_indices_in_full_df
        ]

        # Add 'Top_Risk_Factors' to the columns to display
        existing_display_cols.append("Top_Risk_Factors")
    else:
        st.warning(
            "Feature explanations are not available due to SHAP calculation error."
        )
    # --- END NEW ADDITION ---

    st.dataframe(
        filtered_overdue_predictions_df[existing_display_cols], use_container_width=True
    )
    st.write(f"Total bills flagged as overdue: {len(filtered_overdue_predictions_df)}")
    st.info("Staff can use this list to prioritize proactive outreach.")

    # --- ADD THIS SECTION FOR THE DOWNLOAD BUTTON ---
    # Pass the FULL filtered_overdue_predictions_df to the conversion function
    csv_data = convert_df_to_csv(filtered_overdue_predictions_df)

    st.download_button(
        label="Download Full Flagged Bills List as CSV",  # Text displayed on the button
        data=csv_data,
        file_name="moqhaka_flagged_bills.csv",  # Name of the downloaded file
        mime="text/csv",
        help="Click to download the complete list of bills predicted as overdue.",
    )

else:
    st.info(
        "No bills predicted as overdue at the current threshold and filters. Try adjusting the threshold or filters."
    )


st.markdown("---")

# Section for individual SHAP plots
st.markdown("---")
st.header("🔍 Individual Bill Explanation (SHAP Deep Dive)")
st.write(
    "Select a Bill ID from the table above to understand the key factors influencing its overdue prediction."
)

# --- FIX: Use filtered_overdue_predictions_df for the selectbox ---
if not filtered_overdue_predictions_df.empty:
    selected_bill_id = st.selectbox(
        "Select Bill ID:", filtered_overdue_predictions_df["bill_id"].unique()
    )

    if selected_bill_id:
        # Get the row for the selected bill from the original, full data_for_prediction
        # This is because selected_bill_row will be used to get original_index,
        # which needs to align with data_for_prediction.
        selected_bill_row = data_for_prediction[
            data_for_prediction["bill_id"] == selected_bill_id
        ].iloc[0]

        # Get the corresponding SHAP values and original features for this bill
        # which aligns with the shap_values_df array.
        original_index = selected_bill_row.name
        shap_index_in_array = data_for_prediction.index.get_loc(original_index)

        if shap_values_df is not None:
            st.subheader(f"SHAP Waterfall Plot for Bill ID: {selected_bill_id}")

            # Create a SHAP explainer (can be re-used from load_model_and_scaler if passed)
            explainer = shap.TreeExplainer(model)

            # Retrieve the SHAP values and feature values for the selected instance
            # CORRECTED LINE: Use array indexing for shap_values_df (since it's a NumPy array)
            shap_values_for_instance = shap_values_df[shap_index_in_array]

            # Keep .iloc for X_scaled_for_shap, assuming it's a Pandas DataFrame
            feature_values_for_instance = X_scaled_for_shap.iloc[shap_index_in_array]
            # Get the list of feature names (assuming X_scaled_for_shap is a DataFrame)

            # Get the list of feature names (this assumes X_scaled_for_shap is a DataFrame)
            feature_names_list = X_scaled_for_shap.columns.tolist()

            # Create the shap.Explanation object for this specific instance
            explanation_for_waterfall = shap.Explanation(
                values=shap_values_for_instance,
                base_values=explainer.expected_value,  # This comes from your explainer
                data=feature_values_for_instance.values,  # Pass the raw feature values as a numpy array
                feature_names=feature_names_list,
            )
            fig_shap, ax_shap = plt.subplots(figsize=(10, 7))

            # Plot the waterfall
            # We need to capture the matplotlib figure from SHAP's plot function
            # and then pass it to st.pyplot.
            # To prevent SHAP from immediately showing the plot, set show=False
            shap.waterfall_plot(
                explanation_for_waterfall,  # Pass the Explanation object here
                max_display=15,  # Re-introduce max_display
                show=False,  # Re-introduce show=False for Streamlit
            )

            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            st.pyplot(fig_shap)  # Display the plot in Streamlit
            plt.close(fig_shap)  # Close the figure to free memory

            st.write(
                f"**Base Value (Expected Model Output):** {explainer.expected_value:.2f}"
            )
            st.write(
                f"**Prediction for this bill:** {selected_bill_row['predicted_probability_overdue']:.2f}"
            )
            st.info(
                "This plot shows how each feature contributes to pushing the bill's probability "
                "from the average (base value) to its final prediction. Red indicates a positive "
                "(increases overdue likelihood) contribution, blue indicates negative."
            )
        else:
            st.warning(
                "SHAP values were not computed, so individual explanations are unavailable."
            )
else:
    st.info(
        "Select a threshold that flags bills as overdue and apply filters to see individual explanations."
    )

# --- Section 3: Model Insights - Feature Importance ---
st.header("💡 Model Insights: Top Contributing Factors")
st.write("Understanding which factors most influence the model's predictions.")

# Get feature importance and ensure feature names are aligned
# Note: For XGBoost, model.feature_names_in_ is the most reliable way to get feature names
if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
    feature_names_for_importance = model.feature_names_in_
elif hasattr(model, "feature_names") and model.feature_names is not None:
    feature_names_for_importance = model.feature_names
else:
    # Fallback: Use the FEATURE_COLUMNS defined for data loading in the app
    # This should now be accurate given our previous corrections
    feature_names_for_importance = FEATURE_COLUMNS

feature_importances = model.feature_importances_

if feature_names_for_importance is not None and len(
    feature_names_for_importance
) == len(feature_importances):
    importance_df = pd.DataFrame(
        {"Feature": feature_names_for_importance, "Importance": feature_importances}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df.head(10),
        ax=ax_fi,
        palette="viridis",
    )
    ax_fi.set_title("Top 10 Feature Importances")
    ax_fi.set_xlabel("Importance Score")
    ax_fi.set_ylabel("Feature")
    st.pyplot(fig_fi)
else:
    st.warning(
        "Could not display feature importances. Ensure feature names are correctly retrieved from the model or provided and match the importance array length."
    )

st.markdown("---")

# --- Section 4: Probability Distribution ---
st.header("📈 Predicted Probability Distribution")
st.write("Distribution of predicted probabilities across all bills.")

fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
sns.histplot(
    data_for_prediction["predicted_probability_overdue"], bins=50, kde=True, ax=ax_prob
)
ax_prob.axvline(
    prediction_threshold,
    color="red",
    linestyle="--",
    label=f"Threshold: {prediction_threshold:.2f}",
)
ax_prob.set_title("Distribution of Predicted Probabilities")
ax_prob.set_xlabel("Predicted Probability of Being Overdue")
ax_prob.set_ylabel("Number of Bills")
ax_prob.legend()
st.pyplot(fig_prob)

st.markdown("---")
st.success("Prototype dashboard ready!")

# --- END OF APP.PY ---
