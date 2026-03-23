# Noxim Execution Guide for FL-RIS

This guide contains the exact step-by-step terminal commands needed to apply the Torus patch, recompile Noxim, run the automated IEEE TWC experiments, and generate the final comparison plots on your Ubuntu machine.

## Prerequisites Check
Ensure you have copied the `noxim_configs/`, `noxim_scripts/`, and `noxim_patch/` folders from your Windows machine to your Ubuntu machine. 

For these instructions, we will assume you placed them in a folder called `~/fl_ris_globecom/` in your Ubuntu home directory.

---

## Step 1: Locate your Noxim installation
First, verify where Noxim is installed on your Ubuntu machine. Usually, it's in your home directory (`~/noxim`).

1. Open your Ubuntu terminal.
2. Run this command to verify Noxim is there:
   ```bash
   ls ~/noxim
   ```
   *(If you see folders like `bin`, `src`, `doc`, then this is the right place. If it's installed elsewhere, adjust the paths in the following steps accordingly.)*

---

## Step 2: Test that Noxim works (Baseline Check)
Before changing any code, let's make sure Noxim runs normally.

1. Navigate to the Noxim binary folder:
   ```bash
   cd ~/noxim/bin
   ```
2. Run a simple 4x4 Mesh test with random traffic for 10,000 cycles:
   ```bash
   ./noxim -dimx 4 -dimy 4 -traffic random -sim 10000
   ```
   *(You should see an output block showing "Total received packets", "Total energy (J)", etc. If you see this, Noxim is working perfectly.)*

---

## Step 3: Apply the Torus Topology Patch (Crucial for the Paper)
Your Python experiments proved that the **Torus** topology gives the best results. Noxim doesn't have Torus built-in, so we need to add the code we generated into the Noxim source.

1. Open the patch file we created to read the instructions:
   ```bash
   cat ~/fl_ris_globecom/noxim_patch/torus_topology_patch.cpp
   ```
2. Follow the 10 manual steps listed inside that file. For each step, open the target Noxim source file using a text editor (like `nano` or `gedit`) and paste the code exactly as instructed.
   
   *Example for Step 1:*
   ```bash
   nano ~/noxim/src/GlobalParams.h
   ```
   *(Add `#define TOPOLOGY_TORUS "TORUS"` right below `#define TOPOLOGY_MESH "MESH"`, then save and exit.)*
   
   *For Steps 5 & 6*, you will need to create two entirely new files:
   ```bash
   nano ~/noxim/src/routingAlgorithms/Routing_TORUS_XY.h
   nano ~/noxim/src/routingAlgorithms/Routing_TORUS_XY.cpp
   ```
   *(Paste the code from the patch guide into these new files.)*

---

## Step 4: Recompile Noxim
After you have applied all 10 steps of the patch, you must recompile Noxim so it includes your new Torus C++ code.

1. Navigate back to the Noxim binary compiling folder:
   ```bash
   cd ~/noxim/bin
   ```
2. Clean the old build and compile the new one:
   ```bash
   make clean
   make
   ```
   *(This will output a lot of compilation messages. Wait until it finishes. If there are no errors, the compilation was successful.)*

---

## Step 5: Verify the Torus Patch
Let's make sure the Torus topology actually works now.

1. Run Noxim using the new Torus topology you just added:
   ```bash
   cd ~/noxim/bin
   ./noxim -dimx 4 -dimy 4 -topology TORUS -routing TORUS_XY -traffic random -sim 10000
   ```
   *(If it runs successfully and shows the results block without crashing or throwing an "Unknown topology" error, the patch worked!)*

---

## Step 6: Run Your FL-RIS Experiments
We will now use the automated bash script to run all 8 sets of experiments (A through H) using the optimal baseline configurations and traffic tables.

1. Navigate to the folder where you copied our scripts:
   ```bash
   cd ~/fl_ris_globecom/noxim_scripts/
   ```
2. Make the bash script executable:
   ```bash
   chmod +x run_all_noxim.sh
   ```
3. Run the master script. You must pass the path to your compiled Noxim binary as an argument:
   ```bash
   ./run_all_noxim.sh ~/noxim/bin/noxim
   ```
   *(This will take a little while. It will automatically run Noxim dozens of times using the various YAML files and traffic tables we generated. The output `.txt` files for each run will be saved in `~/fl_ris_globecom/results/noxim_results/`.)*

---

## Step 7: Parse the Results
Raw Noxim output is hard to read. We need to extract the exact latency, throughput, and energy numbers.

1. Run the python parser script (ensure you have `python3` installed on Ubuntu):
   ```bash
   python3 parse_noxim_output.py ~/fl_ris_globecom/results/noxim_results/
   ```
   *(This script will read every `.txt` file, extract the exact numbers, print a clean summary table to your terminal, and generate `noxim_results_parsed.json` and `noxim_results_parsed.csv`.)*

---

## Step 8: Generate the Comparison Plots
Finally, create the beautiful plots comparing your Python analytical math to Noxim's cycle-accurate simulation for your IEEE TWC paper.

1. Ensure you have the `matplotlib` library installed on your Ubuntu machine:
   ```bash
   pip3 install matplotlib
   ```
2. Run the comparison plotter:
   ```bash
   python3 compare_python_noxim.py ~/fl_ris_globecom/results/noxim_results/
   ```
   *(This will generate PDF and PNG files—e.g., `topology_comparison_python_vs_noxim.pdf`—directly inside the `noxim_results` folder.)*

---

## Final Step: Write the Paper
Once you are done, copy the `.csv` files and the newly generated plots back to your Windows machine. We can then directly inject those cycle-accurate, validated numbers into your LaTeX paper to meet the high standards of IEEE Transactions on Wireless Communications!
