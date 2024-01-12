import math
import random

import pandas as pd

# Define the number of entries to generate
num_entries = 5000

# Define the range of temperature values
min_temperature = 10.2
max_temperature = 30.3

# Define the time interval between measurements (30 seconds)
time_interval = 30

# Define the range of expected robot deployment counts
min_expected_robots = 1
max_expected_robots = 4

# Define the range of actual robot presence counts
min_actual_robots = 0
max_actual_robots = max_expected_robots

# Define the range of robot goal fulfillment times (60 to 300 seconds)
min_goal_fulfillment_time = 60
max_goal_fulfillment_time = 300

# Initialize an empty DataFrame to store the generated data
data = []

# Generate data for each entry
for _ in range(num_entries):
    # Generate temperature measurements for 10 intervals
    temperature_measurements = [round(random.uniform(
        min_temperature, max_temperature), 2) for _ in range(10)]

    # Generate robot deployment and presence information
    expected_robots = random.randint(min_expected_robots, max_expected_robots)
    actual_robots = random.randint(min_actual_robots, expected_robots)
    goal_fulfillment_times = []

    if actual_robots < 1:
        goal_fulfillment_times = [0]
    else:
        """ goal_fulfillment_times = [round(random.uniform(
            float(min_goal_fulfillment_time), float(max_goal_fulfillment_time)), 2) for _ in range(actual_robots)] """
        goal_fulfillment_times = [random.randint(
            float(min_goal_fulfillment_time), float(max_goal_fulfillment_time)) for _ in range(actual_robots)]

    median_time = 0
    for time in goal_fulfillment_times:
        median_time += time
    median_time = median_time / len(goal_fulfillment_times)
    median_time = math.floor(median_time)

    # Create a dictionary for the current entry
    entry_data = {
        'Section': random.choice('ABCD'),
        'Temperature Measurements': temperature_measurements,
        'Expected Robots': float(expected_robots),
        'Actual Robots': float(actual_robots),
        'Goal Fulfilment Times': median_time
    }

    # Add the entry to the data list
    data.append(entry_data)

# Create a DataFrame from the generated data
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('warehouse_data.csv', index=False, sep='\t', lineterminator='\n')
