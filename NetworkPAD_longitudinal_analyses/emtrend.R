library(lmerTest)
library(emmeans)
library(tidyr)

setwd("/Users/hyx/Downloads/hyx/Network-based_Brain_Ageing_ADNI/")
data <- read.csv('./adni_BA/lme_data_3group_6yr.csv', header = TRUE)

features <- c('VIS', 'SM', 'DAN', 'VAN', 'LIM', 'FP', 'DMN')

anova_results <- list()
pairwise_results <- list()
slopes_matrix <- data.frame() # 用于存储斜率

for (feature in features) {
  formula1 <- as.formula(paste(feature, "~ Time + Group + Age + Gender + education + APOE4 + Group:Time + (Time | Subject.ID)"))
  formula2 <- as.formula(paste(feature, "~ Time + Group + Age + Gender + education + APOE4 + (Time | Subject.ID)"))
  
  # 使用 bobyqa 优化器
  control <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  model1 <- lmer(formula1, data = data, control = control)
  model2 <- lmer(formula2, data = data, control = control)
  
  # Perform ANOVA to compare models
  cat("\nFeature:", feature, "\n")
  anova_res <- anova(model1, model2)
  print(anova_res)
  anova_results[[feature]] <- as.data.frame(anova_res)
  
  # Calculate marginal slopes using emtrends
  trend_res <- emtrends(model1, specs = "Group", var = "Time")
  trend_summary <- as.data.frame(summary(trend_res))
  
  trend_summary <- trend_summary[, c("Group", "Time.trend")]
  colnames(trend_summary)[2] <- feature
  
  if (nrow(slopes_matrix) == 0) {
    slopes_matrix <- trend_summary
  } else {
    slopes_matrix <- merge(slopes_matrix, trend_summary, by = "Group", all = TRUE)
  }
  
  # If the difference between models is significant
  # if (anova_res$`Pr(>Chisq)`[2] < 0.05) {
  if (1) {
    # Print the results for review
    cat("\nFeature:", feature, "\n")
    print(summary(trend_res))
    
    # 斜率的组间配对比较
    # 设置 Group 的比较顺序
    # trend_res@grid$Group <- factor(trend_res@grid$Group, levels = c("CN_to_MCI", "MCI_to_AD", "CN_to_CN", "MCI_to_MCI", "AD_to_AD"))
    trend_res@grid$Group <- factor(trend_res@grid$Group, levels = c("CN_to_CN", "MCI_to_MCI", "MCI_to_AD" ))
    pairwise_comp <- pairs(trend_res)
    
    pairwise_summary <- as.data.frame(summary(pairwise_comp))
    pairwise_summary <- pairwise_summary[, c("contrast", "t.ratio", "p.value")]
    pairwise_summary$Feature <- feature
    pairwise_results[[feature]] <- pairwise_summary
  }
  
}
all_pairwise_results <- do.call(rbind, pairwise_results)
wide_table <- pivot_wider(
  all_pairwise_results,
  names_from = Feature, #
  values_from = c(t.ratio, p.value), # 包含 t.ratio 和 p.value
  names_glue = "{Feature}_{.value}"  # 
)
write.csv(all_pairwise_results, "adni_BA1/fu_pairwise_comparisons_6yr.csv", row.names = FALSE)

write.csv(slopes_matrix, "adni_BA1/fu_slopes_results_6yr.csv", row.names = FALSE)

anova_results_df <- do.call(rbind, lapply(names(anova_results), function(name) {
  data.frame(Feature = name, anova_results[[name]])
}))
write.csv(anova_results_df, "adni_BA1/lme_fu_anova_results_6yr.csv", row.names = FALSE)