with open("utils/plotting.py", "r") as f:
    content = f.read()

content = content.replace('"""\nPlotting utilities for single FL training runs and basic evaluations.\nFor cross-experiment comparison plots, see `plotting_advanced.py`.\n"""\n\n"""\nPublication-Quality Plotting for FL-RIS Research\n=================================================\n',
'"""\nPublication-Quality Plotting for FL-RIS Research (Base Utilities)\n=================================================\nPlotting utilities for single FL training runs and basic evaluations.\nFor cross-experiment comparison plots, see `plotting_advanced.py`.\n\n')

with open("utils/plotting.py", "w") as f:
    f.write(content)
