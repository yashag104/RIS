with open("utils/plotting_advanced.py", "r") as f:
    content = f.read()

content = content.replace('Advanced Plotting Functions for FL-RIS Research Experiments', 'Cross-Experiment Comparison Plotting Functions')

with open("utils/plotting_advanced.py", "w") as f:
    f.write(content)
