library(lmerTest)
library(emmeans)
library(tidyr)

setwd("/Users/hyx/Downloads/hyx/Network-based_Brain_Ageing_ADNI/")
data <- read.csv('./cluster/bl_4group_6yr.csv', header = TRUE)
features <- c('VIS', 'SM', 'DAN', 'VAN', 'LIM', 'FP', 'DMN')

anova_results <- data.frame()
pairwise_results <- list()
emmean_matrix <- data.frame() # 用于边际均值

anova_output_file <- "adni_BA1/bl_anova_results_6yr.csv"
emmeans_output_file <- "adni_BA1/bl_emmeans_results_6yr.csv"
pairwise_output_file <- "adni_BA1/bl_pairwise_comparisons_6yr.csv"

for (feature in features) {
  formula <- as.formula(paste(feature, "~ Group + Age + Gender + education + APOE4"))
  model <- lm(formula, data = data)
  
  anova_result <- anova(model)
  group_row <- anova_result["Group", c("F value", "Pr(>F)")]
  group_row$Feature <- feature
  anova_results <- rbind(anova_results, group_row)
  
  # Marginal mean and inter group comparison
  emmeans_result <- emmeans(model, "Group")
  emmeans_df <- as.data.frame(emmeans_result)
  
  emmeans_df <- emmeans_df[, c("Group", "emmean")]
  colnames(emmeans_df)[2] <- feature
  
  if (nrow(emmean_matrix) == 0) {
    emmean_matrix <- emmeans_df
  } else {
    emmean_matrix <- merge(emmean_matrix, emmeans_df, by = "Group", all = TRUE)
  }
  
  emmeans_result@grid$Group <- factor(emmeans_result@grid$Group, levels = c("CN", "MCI_to_MCI",  "MCI_to_AD",  "AD"))
  pairwise_comp <- pairs(emmeans_result)
  
  pairwise_summary <- as.data.frame(summary(pairwise_comp))
  pairwise_summary <- pairwise_summary[, c("contrast", "t.ratio", "p.value")]  # df=975
  pairwise_summary$Feature <- feature
  pairwise_results[[feature]] <- pairwise_summary
  
  cat("Results for", feature, ":\n")
  print(anova_result)
  print(emmeans_result)
  print(pairwise_results)
}
all_pairwise_results <- do.call(rbind, pairwise_results)
wide_table <- pivot_wider(
  all_pairwise_results,
  names_from = Feature, 
  values_from = c(t.ratio, p.value), 
  names_glue = "{Feature}_{.value}"  
)

anova_results <- anova_results[, c("Feature", "F value", "Pr(>F)")]

write.csv(all_pairwise_results, pairwise_output_file, row.names = FALSE)
write.csv(anova_results, anova_output_file, row.names = FALSE)
write.csv(emmean_matrix, emmeans_output_file, row.names = FALSE)
