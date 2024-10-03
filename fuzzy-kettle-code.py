import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class Kettle_Fuzzy:
    def __init__(self, current_temp, desired_temp):
        self.current_temp = current_temp
        self.desired_temp = desired_temp

        # Define the universe of discourse for temperature and heating power
        self.temp_set = np.arange(0, 101, 1)  # Temperature input from 0 to 100°C
        self.power_set = np.arange(0, 101, 1)  # Heating power output from 0 to 100%

        # Fuzzy membership functions for current temperature
        self.current_temp_low = fuzz.trimf(self.temp_set, [0, 0, 50])
        self.current_temp_medium = fuzz.trimf(self.temp_set, [25, 50, 75])
        self.current_temp_high = fuzz.trimf(self.temp_set, [50, 100, 100])

        # Fuzzy membership functions for desired temperature
        self.desired_temp_low = fuzz.trimf(self.temp_set, [0, 0, 50])
        self.desired_temp_medium = fuzz.trimf(self.temp_set, [25, 50, 75])
        self.desired_temp_high = fuzz.trimf(self.temp_set, [50, 100, 100])

        # Fuzzy membership functions for heating power
        self.power_low = fuzz.trimf(self.power_set, [0, 0, 50])
        self.power_medium = fuzz.trimf(self.power_set, [25, 50, 75])
        self.power_high = fuzz.trimf(self.power_set, [50, 100, 100])

    # Fuzzy inference rules
    def fuzzy_rule(self):
        # Current temperature membership
        current_low = fuzz.interp_membership(self.temp_set, self.current_temp_low, self.current_temp)
        current_medium = fuzz.interp_membership(self.temp_set, self.current_temp_medium, self.current_temp)
        current_high = fuzz.interp_membership(self.temp_set, self.current_temp_high, self.current_temp)

        # Desired temperature membership
        desired_low = fuzz.interp_membership(self.temp_set, self.desired_temp_low, self.desired_temp)
        desired_medium = fuzz.interp_membership(self.temp_set, self.desired_temp_medium, self.desired_temp)
        desired_high = fuzz.interp_membership(self.temp_set, self.desired_temp_high, self.desired_temp)

        # Rules
        rule1 = np.fmin(current_low, desired_high) #PH
        rule2 = np.fmin(current_medium, desired_high) #PM
        rule3 = np.fmin(current_high, desired_high) #PL
        rule4 = np.fmin(current_low, desired_medium) #PM
        rule5 = np.fmin(current_medium, desired_medium) #PL
        rule6 = np.fmin(current_low, desired_low) #PL
        rule7 = np.fmin(current_high, desired_medium) #PL
        rule8 = np.fmin(current_medium, desired_low) #PL

        # Output power aggregation
        power_activation_low = np.fmin(rule1, self.power_high)
        power_activation_medium = np.fmin(rule2, self.power_medium)
        power_activation_high = np.fmin(rule3, self.power_low)
        power_activation_low_2 = np.fmin(rule4, self.power_medium)
        power_activation_high_2 = np.fmin(rule5, self.power_low)
        power_activation_high_3 = np.fmin(rule6, self.power_low)
        power_activation_high_4 = np.fmin(rule7, self.power_low)
        power_activation_high_5 = np.fmin(rule8, self.power_low)

        # Combine all outputs
        aggregated = np.fmax(power_activation_low, np.fmax(power_activation_medium, np.fmax(power_activation_high,
                       np.fmax(power_activation_low_2, np.fmax(power_activation_high_2, np.fmax(power_activation_high_3,
                       np.fmax(power_activation_high_4, power_activation_high_5)))))))

        return aggregated

    # Defuzzification: Centroid method to get crisp value
    def defuzzify(self, output_fuzzy):
        return np.sum(output_fuzzy * self.power_set) / np.sum(output_fuzzy)

    # Control system function: Input = current and desired temperature, Output = heating power
    def fuzzy_control(self):
        # Apply fuzzy rules
        fuzzy_output = self.fuzzy_rule()
        
        # Defuzzify the fuzzy output to get a crisp heating power
        heating_power = self.defuzzify(fuzzy_output)
        
        return heating_power

    # Visualizing the fuzzy sets and output
    def plot_fuzzy_control(self):
        plt.figure(figsize=(10, 8))
        
        # Plot membership functions for current temperature
        plt.subplot(3, 1, 1)
        plt.plot(self.temp_set, self.current_temp_low, label='Low')
        plt.plot(self.temp_set, self.current_temp_medium, label='Medium')
        plt.plot(self.temp_set, self.current_temp_high, label='High')
        plt.axvline(x=self.current_temp, color='black', linestyle='--', label=f'Current Temp = {self.current_temp}°C')
        plt.title('Fuzzy Set: Current Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid()

        # Plot membership functions for desired temperature
        plt.subplot(3, 1, 2)
        plt.plot(self.temp_set, self.desired_temp_low, label='Low')
        plt.plot(self.temp_set, self.desired_temp_medium, label='Medium')
        plt.plot(self.temp_set, self.desired_temp_high, label='High')
        plt.axvline(x=self.desired_temp, color='black', linestyle='--', label=f'Desired Temp = {self.desired_temp}°C')
        plt.title('Fuzzy Set: Desired Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Membership Degree')
        plt.legend()
        plt.grid()

        # Plot the result of fuzzy inference for heating power
        plt.subplot(3, 1, 3)
        fuzzy_output = self.fuzzy_rule()
        plt.plot(self.power_set, fuzzy_output, label='Fuzzy Output')
        plt.fill_between(self.power_set, 0, fuzzy_output, alpha=0.3)
        plt.axvline(x=self.fuzzy_control(), color='black', linestyle='--', label=f'Heating Power = {self.fuzzy_control():.2f}%')
        plt.title(f'Fuzzy Output for Current Temp = {self.current_temp}°C and Desired Temp = {self.desired_temp}°C')
        plt.xlabel('Heating Power (%)')
        plt.ylabel('Membership Degree')
        plt.grid()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Set fixed input values for current and desired temperature
    current_temp_input = 50  # Change this value as needed
    desired_temp_input = 20  # Change this value as needed
    kettle = Kettle_Fuzzy(current_temp_input, desired_temp_input)
    heating_power = kettle.fuzzy_control()
    print(f'For current temperature = {current_temp_input} C and desired temperature = {desired_temp_input} C, the heating power is: {heating_power:.2f}%')

    # Plot the fuzzy control system for the given temperature inputs
    kettle.plot_fuzzy_control()
