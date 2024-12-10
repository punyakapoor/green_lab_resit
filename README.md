LLM Energy Consumption Experiment
This repository contains the implementation, data, and analysis for a study on the impact of prompt types on the energy consumption of Large Language Models (LLMs) during inference. The goal is to benchmark energy efficiency, resource utilization, and response time across different LLMs, prompt types, and input lengths.

Project Structure
Data
results.csv: Consolidated metrics collected during the experiment.
results.json: Raw data from all experimental runs.
results.xlsx: An Excel version of the results for easier analysis.
Data Analysis
results/: Contains intermediate and processed data files.
distribution_analysis.csv: Statistical summaries of variable distributions.
enhanced_descriptive_stats.csv: Detailed descriptive statistics for each metric.
kruskal_results.csv: Results of the Kruskal-Wallis tests for non-parametric analysis.
pairwise_effects.csv: Effect sizes for pairwise comparisons between groups.
data_analysis.r: R script for statistical analysis and visualization.
requirements: List of dependencies for running the R script.
Experiment Code
Experiment.py: Python script to run the experiment pipeline, including LLM inference, metric collection, and logging.
Visualization
plots/: Contains all visualizations generated from the experiment.
cpu_vs_gpu_utilization.png: CPU and GPU utilization trends.
effect_sizes.png: Effect sizes for energy consumption comparisons.
energy_consumption_boxplot.png: Boxplot showing energy consumption by model and prompt type.
gpu_utilization_lineplot.png: GPU utilization over multiple runs.
gpu_vs_cpu_power.png: Scatter plot comparing GPU and CPU power consumption.
power_distribution.png: Distribution of power usage across models and prompt types.
qq_energy.png: Q-Q plot for energy consumption residuals.
qq_response.png: Q-Q plot for response time residuals.
Requirements
Python Dependencies
Python 3.11
Required libraries: See requirements.txt in the repository.
R Dependencies
Libraries: tidyverse, ggplot2, dplyr, car, psych, corrplot, MVN, etc.
How to Run the Experiment
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo-name.git
cd your-repo-name
Run the Python experiment script:

bash
Copy code
python Experiment_Code/Experiment.py
Perform statistical analysis using the R script:

bash
Copy code
Rscript Data_Analysis/data_analysis.r
View results:

Raw and processed data: Located in the Data/ and Data_Analysis/results/ directories.
Visualizations: Found in Visualisation/plots/.
Results Summary
The experiment investigated the influence of prompt types on energy consumption and resource utilization of LLMs:

Energy Consumption: Summarization prompts exhibited the highest energy demands across all models.
Response Time: Summarization tasks had the longest inference times, followed by sentiment analysis and classification.
Resource Utilization: GPU utilization was significantly higher for complex tasks.
For detailed results, refer to the final report in the report/ directory.

