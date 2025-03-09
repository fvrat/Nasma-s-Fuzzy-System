import numpy as np
from scipy.signal import find_peaks
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import uuid
import os

# Ensure Firebase credentials are set securely

# Function to fetch patient's Date of Birth from Firebase and calculate age
def get_patient_age(patient_id):
    ref = db.reference(f"/Patient/{patient_id}/Date_of_birth")
    dob_str = ref.get()
    if dob_str:
        dob = datetime.strptime(dob_str, "%d/%m/%Y")
        today = datetime.today()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    return None

# Function to send alert to Firebase
def send_alert_to_firebase(patient_id, alert_message, reason):
    alert_ref = db.reference(f"/Patient/{patient_id}/Alert")
    alert_ref.child(str(uuid.uuid4())).set({
        "message": alert_message,
        "reason": reason,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Function to calculate Respiratory Rate from accelerometer data
def calculate_respiratory_rate(accelerometer_data, sampling_rate=50):
    smoothed_data = np.convolve(accelerometer_data, np.ones(5)/5, mode='valid')
    peaks, _ = find_peaks(smoothed_data, distance=sampling_rate/2)
    duration_in_seconds = len(accelerometer_data) / sampling_rate
    breaths_per_minute = (len(peaks) / duration_in_seconds) * 60
    return breaths_per_minute

# Function to calculate Heart Rate from PPG signal
def calculate_heart_rate(ppg_signal, sampling_rate=50):
    peaks, _ = find_peaks(ppg_signal, distance=sampling_rate/2)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / sampling_rate
        hr = 60 / np.mean(rr_intervals)
    else:
        hr = np.random.randint(50, 150)  # Fallback value
    return hr

# Function to calculate Oxygen Saturation from PPG signal
def calculate_oxygen_saturation(ppg_red, ppg_ir):
    ac_red = np.ptp(ppg_red)
    dc_red = np.mean(ppg_red)
    ac_ir = np.ptp(ppg_ir)
    dc_ir = np.mean(ppg_ir)
    spo2 = 110 - 25 * (ac_red / dc_red) / (ac_ir / dc_ir)
    return np.clip(spo2, 85, 100)

# Define function to determine patient category
def get_patient_category(age):
    if age is None:
        return 'unknown'
    elif age < 1:
        return 'infant'
    elif age < 5:
        return 'preschooler'
    elif age < 13:
        return 'school_age'
    else:
        return 'adult'

# Define patient-specific thresholds using exact values from tables
def get_thresholds(category):
    if category == 'infant':
        return {'oxygen_saturation': 94, 'respiratory_rate': 60, 'heart_rate_low': 60, 'heart_rate_high': 120}
    elif category == 'preschooler':
        return {'oxygen_saturation': 94, 'respiratory_rate': 40, 'heart_rate_low': 60, 'heart_rate_high': 110}
    elif category == 'school_age':
        return {'oxygen_saturation': 94, 'respiratory_rate': 30, 'heart_rate_low': 60, 'heart_rate_high': 100}
    elif category == 'adult':
        return {'oxygen_saturation': 92, 'respiratory_rate': 25, 'heart_rate_low': 60, 'heart_rate_high': 100}
    return None

# Define patient ID
patient_id = "1"  # Change dynamically if needed

# Fetch patient age from Firebase
patient_age = get_patient_age(patient_id)
category = get_patient_category(patient_age)
thresh = get_thresholds(category)

if category == 'unknown' or not thresh:
    print("Error: Unable to determine patient category or thresholds.")
else:
    # Simulated Data
    respiratory_rate_value = np.random.uniform(10, 60)
    oxygen_saturation_value = np.random.uniform(85, 100)
    heart_rate_value = np.random.uniform(50, 150)

    # Check if any value is in high-priority range and log reason
    reasons = []
    if oxygen_saturation_value < thresh['oxygen_saturation']:
        reasons.append(f"Low Oxygen Saturation: {oxygen_saturation_value:.2f}% (<{thresh['oxygen_saturation']}%)")
    if respiratory_rate_value > thresh['respiratory_rate']:
        reasons.append(f"High Respiratory Rate: {respiratory_rate_value:.2f} bpm (>{thresh['respiratory_rate']} bpm)")
    if heart_rate_value < thresh['heart_rate_low'] or heart_rate_value > thresh['heart_rate_high']:
        reasons.append(f"Abnormal Heart Rate: {heart_rate_value:.2f} bpm (<{thresh['heart_rate_low']} bpm or >{thresh['heart_rate_high']} bpm)")

    if reasons:
        alert_message = "ALERT: Seek Emergency Care (ER) Immediately!"
        reason_text = " | ".join(reasons)
        print(alert_message, reason_text)
        send_alert_to_firebase(patient_id, alert_message, reason_text)

    print(f"Category: {category.capitalize()}")
    print(f"Oxygen Saturation: {oxygen_saturation_value:.2f}% (High Alert if <{thresh['oxygen_saturation']}%)")
    print(f"Respiratory Rate: {respiratory_rate_value:.2f} bpm (High Alert if >{thresh['respiratory_rate']} bpm)")
    print(f"Heart Rate: {heart_rate_value:.2f} bpm (High Alert if <{thresh['heart_rate_low']} bpm or >{thresh['heart_rate_high']} bpm)")
