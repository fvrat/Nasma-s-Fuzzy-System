import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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

# Define fuzzy variables
oxygen_saturation = ctrl.Antecedent(np.arange(85, 101, 1), 'oxygen_saturation')
respiratory_rate = ctrl.Antecedent(np.arange(10, 60, 1), 'respiratory_rate')
cough_count = ctrl.Antecedent(np.arange(0, 20, 1), 'cough_count')
sleep_interruptions = ctrl.Antecedent(np.arange(0, 10, 1), 'sleep_interruptions')
heart_rate = ctrl.Antecedent(np.arange(50, 150, 1), 'heart_rate')
temperature = ctrl.Antecedent(np.arange(35, 41, 0.1), 'temperature')
status = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'status')

# Membership functions
oxygen_saturation.automf(3)
respiratory_rate.automf(3)
cough_count.automf(3)
sleep_interruptions.automf(3)
heart_rate.automf(3)
temperature.automf(3)
status.automf(3)

# Define rules
rule1 = ctrl.Rule(oxygen_saturation['poor'] | respiratory_rate['poor'] | cough_count['poor'] | sleep_interruptions['poor'] | heart_rate['poor'] | temperature['poor'], status['poor'])
rule2 = ctrl.Rule(oxygen_saturation['good'] & respiratory_rate['good'] & cough_count['good'] & sleep_interruptions['good'] & heart_rate['good'] & temperature['good'], status['good'])
rule3 = ctrl.Rule(oxygen_saturation['average'] | respiratory_rate['average'] | cough_count['average'] | sleep_interruptions['average'] | heart_rate['average'] | temperature['average'], status['average'])

# Control System
status_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
status_simulation = ctrl.ControlSystemSimulation(status_ctrl)

# Simulated Data
time = np.linspace(0, 60, 3000)
simulated_accelerometer = np.sin(2 * np.pi * 0.25 * time) + np.random.normal(0, 0.1, len(time))
simulated_ppg_red = np.sin(2 * np.pi * 1.2 * time) + np.random.normal(0, 0.05, len(time))
simulated_ppg_ir = np.sin(2 * np.pi * 1.1 * time) + np.random.normal(0, 0.05, len(time))

respiratory_rate_value = calculate_respiratory_rate(simulated_accelerometer)
cough_count_value = np.random.randint(0, 15)
sleep_interruptions_value = np.random.randint(0, 10)
oxygen_saturation_value = calculate_oxygen_saturation(simulated_ppg_red, simulated_ppg_ir)
heart_rate_value = calculate_heart_rate(simulated_ppg_ir)
temperature_value = np.random.uniform(35, 41)

# Set inputs
status_simulation.input['oxygen_saturation'] = oxygen_saturation_value
status_simulation.input['respiratory_rate'] = respiratory_rate_value
status_simulation.input['cough_count'] = cough_count_value
status_simulation.input['sleep_interruptions'] = sleep_interruptions_value
status_simulation.input['heart_rate'] = heart_rate_value
status_simulation.input['temperature'] = temperature_value

# Compute output
status_simulation.compute()
final_status = status_simulation.output['status']

# Output
print(f"Status: {final_status:.2f}")
print(f"Oxygen Saturation: {oxygen_saturation_value:.2f}%, Respiratory Rate: {respiratory_rate_value:.2f} bpm, Heart Rate: {heart_rate_value:.2f} bpm, Body Temperature: {temperature_value:.2f}°C")
print(f"Cough Count: {cough_count_value}, Sleep Interruptions: {sleep_interruptions_value}")

# Plotting
plt.figure(figsize=(12, 4))
plt.plot(time, simulated_accelerometer)
plt.title('Simulated Accelerometer Data (Chest Movement)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
