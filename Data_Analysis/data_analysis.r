library(tidyverse)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(car)
library(ggpubr)
library(corrplot)
library(psych)
library(jsonlite)
library(outliers)  
library(moments)   
library(effectsize) 
library(MVN)       

dir.create("results", showWarnings = FALSE)
dir.create("results/plots", showWarnings = FALSE)
dir.create("results/data", showWarnings = FALSE)


tryCatch({
    print(paste("Current working directory:", getwd()))
    data_file <- "/Users/punyakapoor/Documents/experiment-runner/experiment-runner/results.json"
    
    if (!file.exists(data_file)) {
        stop(paste("File not found:", data_file))
    }
    
    print(paste("Reading file:", data_file))
    aggregated_df <- fromJSON(data_file)
    print(paste("Successfully read data. Number of rows:", nrow(aggregated_df)))
    
}, error = function(e) {
    stop(paste("Error reading JSON file:", e))
})

cat("\n--- Data Preprocessing ---\n")

# 1. Initial type conversion and feature creation
aggregated_df <- aggregated_df %>%
  mutate(
    model = factor(model),
    prompt_type = factor(prompt_type),
    input_length = factor(input_length, levels = c("short", "long")),
    energy_consumption = as.numeric(cpu_power + gpu_power),
    response_time = as.numeric(response_time),
    gpu_utilization = as.numeric(gpu_utilization),
    cpu_utilization = as.numeric(cpu_utilization),
    vram_usage = as.numeric(vram_usage),
    cpu_power = as.numeric(cpu_power),
    gpu_power = as.numeric(gpu_power),
    power_ratio = gpu_power / cpu_power,
    utilization_ratio = gpu_utilization / cpu_utilization,
    efficiency = energy_consumption / response_time
  )

# 2. Check for missing values
missing_values <- colSums(is.na(aggregated_df))
cat("\nMissing Values Check:\n")
print(missing_values)

# 3. Outlier Detection
numeric_cols <- c("energy_consumption", "response_time", "gpu_utilization", 
                 "cpu_utilization", "vram_usage", "cpu_power", "gpu_power")

outlier_summary <- data.frame(variable = numeric_cols)
outlier_summary$outliers <- sapply(numeric_cols, function(col) {
    sum(abs(scale(aggregated_df[[col]])) > 3)
})

cat("\nOutlier Summary (|z-score| > 3):\n")
print(outlier_summary)

# 4. Data Validation
validation_summary <- data.frame(
    check = c("GPU Utilization Range", "CPU Utilization Range", 
              "Response Time Positive", "Power Values Positive"),
    result = c(
        all(aggregated_df$gpu_utilization >= 0 & aggregated_df$gpu_utilization <= 100),
        all(aggregated_df$cpu_utilization >= 0 & aggregated_df$cpu_utilization <= 100),
        all(aggregated_df$response_time > 0),
        all(aggregated_df$cpu_power > 0 & aggregated_df$gpu_power > 0)
    )
)

cat("\nData Validation Summary:\n")
print(validation_summary)
cat("\n--- Enhanced Descriptive Statistics ---\n")

# 1. Basic Statistics
basic_stats <- aggregated_df %>%
  group_by(model, prompt_type) %>%
  summarise(across(all_of(numeric_cols), 
                  list(
                    mean = ~mean(., na.rm = TRUE),
                    sd = ~sd(., na.rm = TRUE),
                    median = ~median(., na.rm = TRUE),
                    min = ~min(., na.rm = TRUE),
                    max = ~max(., na.rm = TRUE),
                    q1 = ~quantile(., 0.25, na.rm = TRUE),
                    q3 = ~quantile(., 0.75, na.rm = TRUE),
                    iqr = ~IQR(., na.rm = TRUE),
                    skew = ~skewness(., na.rm = TRUE),
                    kurt = ~kurtosis(., na.rm = TRUE)
                  )),
    .groups = 'drop')

# 2. Distribution Analysis
distribution_stats <- aggregated_df %>%
  summarise(across(all_of(numeric_cols),
                  list(
                    shapiro_p = ~shapiro.test(.)$p.value,
                    normality = ~ifelse(shapiro.test(.)$p.value > 0.05, "Normal", "Non-normal")
                  )))

cat("\nDistribution Analysis:\n")
print(distribution_stats)

# 3. Correlation Analysis with significance
correlation_matrix <- cor(aggregated_df[numeric_cols])
correlation_p_values <- corr.test(aggregated_df[numeric_cols])$p

cat("\nSignificant Correlations (p < 0.05):\n")
significant_correlations <- which(correlation_p_values < 0.05, arr.ind = TRUE)
print(significant_correlations)
cat("\n--- Enhanced Hypothesis Testing ---\n")

# 1. Multivariate Normality Test
cat("\n1. Multivariate Normality Test:\n")
mv_normality <- mvn(aggregated_df[numeric_cols], mvnTest = "mardia")
print(mv_normality$multivariateNormality)

# 2. ANOVA and Effect Size Analysis
cat("\n2. ANOVA Analysis:\n")

# Energy Consumption
energy_anova <- aov(energy_consumption ~ model * prompt_type * input_length, data = aggregated_df)
cat("\nEnergy Consumption ANOVA:\n")
print(summary(energy_anova))
cat("\nEffect Sizes (Partial Eta Squared):\n")
print(eta_squared(energy_anova, partial = TRUE))

# Response Time
response_anova <- aov(response_time ~ model * prompt_type * input_length, data = aggregated_df)
cat("\nResponse Time ANOVA:\n")
print(summary(response_anova))
cat("\nEffect Sizes (Partial Eta Squared):\n")
print(eta_squared(response_anova, partial = TRUE))

# 3. Post-hoc Analysis
cat("\n3. Post-hoc Analysis:\n")

# Tukey's HSD
tukey_energy <- TukeyHSD(energy_anova)
tukey_response <- TukeyHSD(response_anova)

# Function to calculate effect size between two groups
calculate_effect_size <- function(data, var, group1, group2) {
  g1_data <- data[data$prompt_type == group1, var]
  g2_data <- data[data$prompt_type == group2, var]
  
  pooled_sd <- sqrt(((length(g1_data)-1)*var(g1_data) + (length(g2_data)-1)*var(g2_data)) /
                     (length(g1_data) + length(g2_data) - 2))
  d <- abs(mean(g1_data) - mean(g2_data)) / pooled_sd
  return(d)
}

# Calculate pairwise effect sizes
prompt_types <- unique(aggregated_df$prompt_type)
models <- unique(aggregated_df$model)
pairwise_effects <- data.frame()

for (model in models) {
  model_data <- aggregated_df[aggregated_df$model == model,]
  
  for (i in 1:(length(prompt_types)-1)) {
    for (j in (i+1):length(prompt_types)) {
      effect_size_energy <- calculate_effect_size(model_data, "energy_consumption", 
                                                prompt_types[i], prompt_types[j])
      effect_size_response <- calculate_effect_size(model_data, "response_time", 
                                                  prompt_types[i], prompt_types[j])
      
      pairwise_effects <- rbind(pairwise_effects, data.frame(
        model = model,
        comparison = paste(prompt_types[i], "vs", prompt_types[j]),
        cohens_d_energy = effect_size_energy,
        cohens_d_response = effect_size_response
      ))
    }
  }
}

# Interpret effect sizes
pairwise_effects$energy_effect_magnitude <- cut(pairwise_effects$cohens_d_energy,
                                              breaks = c(-Inf, 0.2, 0.5, 0.8, Inf),
                                              labels = c("Negligible", "Small", "Medium", "Large"))

pairwise_effects$response_effect_magnitude <- cut(pairwise_effects$cohens_d_response,
                                                breaks = c(-Inf, 0.2, 0.5, 0.8, Inf),
                                                labels = c("Negligible", "Small", "Medium", "Large"))

cat("\nEffect Size Analysis Results:\n")
print(pairwise_effects)

# 4. Non-parametric Tests
cat("\n4. Non-parametric Analysis:\n")

# Kruskal-Wallis Tests
perform_kruskal <- function(dv, iv, data, test_name) {
  result <- kruskal.test(data[[dv]] ~ data[[iv]])
  return(data.frame(
    test = test_name,
    statistic = result$statistic,
    p_value = result$p.value
  ))
}

kruskal_results <- rbind(
  perform_kruskal("energy_consumption", "model", aggregated_df, "Energy by Model"),
  perform_kruskal("energy_consumption", "prompt_type", aggregated_df, "Energy by Prompt Type"),
  perform_kruskal("response_time", "model", aggregated_df, "Response Time by Model"),
  perform_kruskal("response_time", "prompt_type", aggregated_df, "Response Time by Prompt Type"),
  perform_kruskal("gpu_utilization", "model", aggregated_df, "GPU Utilization by Model"),
  perform_kruskal("cpu_utilization", "model", aggregated_df, "CPU Utilization by Model")
)

cat("\nKruskal-Wallis Test Results:\n")
print(kruskal_results)

# Save statistical results
write.csv(basic_stats, "results/data/enhanced_descriptive_stats.csv", row.names = FALSE)
write.csv(distribution_stats, "results/data/distribution_analysis.csv", row.names = FALSE)
write.csv(pairwise_effects, "results/data/pairwise_effects.csv", row.names = FALSE)
write.csv(kruskal_results, "results/data/kruskal_results.csv", row.names = FALSE)
# Visualization Section

# Statistical Plots
# 1. Effect Size Plot
p_effect_sizes <- ggplot(pairwise_effects, aes(x = comparison, y = cohens_d_energy, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~model) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  ) +
  ggtitle("Effect Sizes for Energy Consumption Between Prompt Types") +
  xlab("Comparison") +
  ylab("Cohen's d")

# 2. Q-Q plots for ANOVA residuals
p_qq_energy <- ggplot(data.frame(residuals = residuals(energy_anova)), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line() +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  ) +
  ggtitle("Q-Q Plot of Energy ANOVA Residuals")

p_qq_response <- ggplot(data.frame(residuals = residuals(response_anova)), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line() +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  ) +
  ggtitle("Q-Q Plot of Response Time ANOVA Residuals")

# Performance Plots
# 1. Line Plot for GPU Utilization Over Runs
p1 <- ggplot(aggregated_df, aes(x = run_nr, y = gpu_utilization, color = prompt_type, linetype = model, group = interaction(prompt_type, model))) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  ggtitle("GPU Utilization Over Runs for Each Prompt Type and Model") +
  xlab("Run Number") +
  ylab("GPU Utilization (%)") +
  theme_minimal() +
  theme(
    legend.position = "top",
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  )

# 2. CPU vs GPU Utilization Line Graph
p2 <- ggplot(aggregated_df, aes(x = run_nr)) +
  geom_line(aes(y = gpu_utilization, color = "GPU"), size = 1) +
  geom_line(aes(y = cpu_utilization, color = "CPU"), size = 1) +
  geom_point(aes(y = gpu_utilization, color = "GPU"), size = 2) +
  geom_point(aes(y = cpu_utilization, color = "CPU"), size = 2) +
  facet_grid(model ~ prompt_type) +
  scale_color_manual(values = c("GPU" = "darkblue", "CPU" = "darkred")) +
  ggtitle("CPU vs GPU Utilization Over Runs") +
  xlab("Run Number") +
  ylab("Utilization (%)") +
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.title = element_blank(),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  )

# 3. Energy Consumption Box Plot
p3 <- ggplot(aggregated_df, aes(x = model, y = energy_consumption, fill = prompt_type)) +
  geom_boxplot() +
  ggtitle("Energy Consumption by Model and Prompt Type") +
  xlab("Model") +
  ylab("Energy Consumption (W)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  )

# 4. Response Time Bar Plot
p4 <- ggplot(aggregated_df, aes(x = model, y = response_time, fill = prompt_type)) +
  stat_summary(fun = mean, geom = "bar", position = "dodge") +
  stat_summary(fun.data = mean_se, geom = "errorbar", position = position_dodge(0.9), width = 0.2) +
  ggtitle("Average Response Time by Model and Prompt Type") +
  xlab("Model") +
  ylab("Response Time (s)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  )

# 5. GPU Power vs CPU Power
p5 <- ggplot(aggregated_df, aes(x = cpu_power, y = gpu_power, color = model, shape = prompt_type)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
  facet_wrap(~model) +
  ggtitle("GPU Power vs CPU Power by Model") +
  xlab("CPU Power (W)") +
  ylab("GPU Power (W)") +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95"),
    legend.position = "top"
  )

# 6. Power Distribution
# Prepare data for power distribution plot
power_summary <- aggregated_df %>%
  group_by(model, prompt_type) %>%
  summarise(
    avg_cpu_power = mean(cpu_power),
    avg_gpu_power = mean(gpu_power),
    .groups = 'drop'
  ) %>%
  pivot_longer(
    cols = c(avg_cpu_power, avg_gpu_power),
    names_to = "power_type",
    values_to = "power"
  )

p6 <- ggplot(power_summary, aes(x = model, y = power, fill = power_type)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(~prompt_type) +
  ggtitle("Power Distribution by Model and Prompt Type") +
  xlab("Model") +
  ylab("Power (W)") +
  scale_fill_manual(
    values = c("avg_cpu_power" = "lightblue", "avg_gpu_power" = "darkblue"),
    labels = c("CPU Power", "GPU Power"),
    name = "Power Type"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_line(color = "grey95")
  )

# Save all plots
tryCatch({
    plot_dir <- "results/plots"
    
    ggsave(file.path(plot_dir, "effect_sizes.png"), p_effect_sizes, width = 12, height = 8, bg = "white")
    ggsave(file.path(plot_dir, "qq_energy.png"), p_qq_energy, width = 8, height = 6, bg = "white")
    ggsave(file.path(plot_dir, "qq_response.png"), p_qq_response, width = 8, height = 6, bg = "white")
    ggsave(file.path(plot_dir, "gpu_utilization_lineplot.png"), p1, width = 12, height = 8, bg = "white")
    ggsave(file.path(plot_dir, "cpu_vs_gpu_utilization.png"), p2, width = 14, height = 10, bg = "white")
    ggsave(file.path(plot_dir, "energy_consumption_boxplot.png"), p3, width = 10, height = 8, bg = "white")
    ggsave(file.path(plot_dir, "response_time_barplot.png"), p4, width = 10, height = 8, bg = "white")
    ggsave(file.path(plot_dir, "gpu_vs_cpu_power.png"), p5, width = 12, height = 8, bg = "white")
    ggsave(file.path(plot_dir, "power_distribution.png"), p6, width = 12, height = 8, bg = "white")
    
    cat("\nAll plots saved successfully in:", plot_dir, "\n")
}, error = function(e) {
    cat("\nError saving plots:", e$message, "\n")
})

cat("\n--- Analysis and Visualization Completed ---\n")
cat("\nResults are organized in the following directories:")
cat("\n- Plots:", "results/plots/")
cat("\n- Data:", "results/data/")
cat("\n")