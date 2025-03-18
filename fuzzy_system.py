import numpy as np
from scipy.signal import find_peaks
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import uuid

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
def send_alert_to_firebase(patient_id, alert_message, risk_level):
    alert_ref = db.reference(f"/Patient/{patient_id}/Alert")
    alert_ref.child(str(uuid.uuid4())).set({
        "message": alert_message,
        "risk_level": risk_level,
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

# Fuzzy Logic System
respiratory_rate = ctrl.Antecedent(np.arange(10, 60, 1), 'respiratory_rate')
oxygen_saturation = ctrl.Antecedent(np.arange(85, 100, 1), 'oxygen_saturation')
heart_rate = ctrl.Antecedent(np.arange(50, 150, 1), 'heart_rate')
risk_level = ctrl.Consequent(np.arange(0, 100, 1), 'risk_level')

# Define Membership Functions
respiratory_rate['low'] = fuzz.trimf(respiratory_rate.universe, [10, 15, 20])
respiratory_rate['normal'] = fuzz.trimf(respiratory_rate.universe, [18, 25, 32])
respiratory_rate['high'] = fuzz.trimf(respiratory_rate.universe, [30, 40, 60])

oxygen_saturation['low'] = fuzz.trimf(oxygen_saturation.universe, [85, 88, 92])
oxygen_saturation['normal'] = fuzz.trimf(oxygen_saturation.universe, [90, 95, 100])

heart_rate['low'] = fuzz.trimf(heart_rate.universe, [50, 60, 70])
heart_rate['normal'] = fuzz.trimf(heart_rate.universe, [65, 80, 100])
heart_rate['high'] = fuzz.trimf(heart_rate.universe, [90, 110, 150])

risk_level['low'] = fuzz.trimf(risk_level.universe, [0, 20, 40])
risk_level['moderate'] = fuzz.trimf(risk_level.universe, [30, 50, 70])
risk_level['high'] = fuzz.trimf(risk_level.universe, [60, 80, 100])

# Define Rules
rule1 = ctrl.Rule(respiratory_rate['high'] | oxygen_saturation['low'] | heart_rate['high'], risk_level['high'])
rule2 = ctrl.Rule(respiratory_rate['normal'] & oxygen_saturation['normal'] & heart_rate['normal'], risk_level['low'])
rule3 = ctrl.Rule(respiratory_rate['low'] | heart_rate['low'], risk_level['moderate'])

# Create and Simulate Fuzzy Controller
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)

# Simulated Patient Data
respiratory_rate_value = np.random.uniform(10, 60)
oxygen_saturation_value = np.random.uniform(85, 100)
heart_rate_value = np.random.uniform(50, 150)

# Set Inputs
risk_simulation.input['respiratory_rate'] = respiratory_rate_value
risk_simulation.input['oxygen_saturation'] = oxygen_saturation_value
risk_simulation.input['heart_rate'] = heart_rate_value

# Compute Risk Level
risk_simulation.compute()
final_risk = risk_simulation.output['risk_level']

# Determine Severity Category
if final_risk < 30:
    severity = "Low"
elif final_risk < 60:
    severity = "Moderate"
elif final_risk < 80:
    severity = "High"
else:
    severity = "Severe"

alert_message = f"Risk Level: {severity}. Seek medical attention if needed."
print(alert_message)
send_alert_to_firebase("1", alert_message, severity)

print(f"Respiratory Rate: {respiratory_rate_value:.2f} bpm")
print(f"Oxygen Saturation: {oxygen_saturation_value:.2f}%")
print(f"Heart Rate: {heart_rate_value:.2f} bpm")
print(f"Final Risk Level: {final_risk:.2f} ({severity})")
