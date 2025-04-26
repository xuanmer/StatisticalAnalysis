'''
Two-step ordinary least squares (OLS) regression models:
biomarker - baseline PAD - cognition
biomarker - PAD change rate - cognition decline rate
All models were adjusted for age, sex, and years of education as covariates.

Group:
MCI_to_MCI    300
MCI_to_AD     157
'''
import pandas as pd
from statsmodels.formula.api import ols
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests


df1 = pd.read_csv('./corr/pad_cog_change_ratio_6yr.csv')[['Subject ID','MMSCORE_rate','MOCA_rate']]
df2 = pd.read_csv('corr/bl_rate_corr_data.csv')   # lme 有组信息的变化率
data = pd.merge(df2, df1, on=['Subject ID'], how='left')

print(data.shape)
print(data['Group'].value_counts())
data['Group'] = pd.Categorical(data['Group'], categories=['MCI_to_MCI', 'MCI_to_AD'], ordered=True)
print(data.columns)

covariates = ['Age', 'Gender', 'education']
# cognitive = ['ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'MMSCORE', 'MOCA', 'CDGLOBAL', 'FAQTOTAL',]
cognitions = ['MMSCORE', 'MOCA']
# cognitions = ['MMSCORE_rate', 'MOCA_rate']
# cognitive = ['ADNI_MEM_rate','ADNI_EF_rate','ADNI_LAN_rate','ADNI_VS_rate','MMSCORE_rate','MOCA_rate','CDGLOBAL_rate','FAQTOTAL_rate']
genetic = ['PHS', 'APOE4']
biomarker = ['ABETA42', 'TAU', 'PTAU', 'HCI',  'INFERIOR_TEMPORAL_SUVR', 'ENTORHINAL_SUVR','TAU_METAROI']
mental_health = ['GDTOTAL', 'NPISCORE']

biomarkers = genetic + biomarker
brain_ages = ['VIS', 'SM', 'DAN', 'VAN', 'LIM', 'FP', 'DMN', ]
# brain_ages = ['VIS_PAD_rate', 'SM_PAD_rate', 'DAN_PAD_rate', 'VAN_PAD_rate', 'LIM_PAD_rate', 'FP_PAD_rate', 'DMN_PAD_rate']

features = biomarkers + brain_ages + cognitions + covariates

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])
print(data)

results = []
bootstrap_samples = 5000
alpha = 0.05

# Bootstrap
def bootstrap_effect(resampled_data, bio, ba, cog):
    try:
        med_model_bs = ols(f'{ba} ~ {bio} + Age + Gender + education', data=resampled_data).fit()
        outcome_model_bs = ols(f'{cog} ~ {bio} + {ba} + Age + Gender + education', data=resampled_data).fit()
        return med_model_bs.params[bio] * outcome_model_bs.params[ba]
    except:
        return None

for bio in biomarkers:
    for ba in brain_ages:
        for cog in cognitions:
            print(f"Analyzing: {bio} → {ba} → {cog}")

            subset_data = data[[bio, ba, cog, 'Age', 'Gender', 'education']].dropna()
            if subset_data.shape[0] < 10:
                print(f"Skipping due to insufficient data: {bio} → {ba} → {cog}")
                continue

            # X → M (biomarker → brain_age)
            med_model = ols(f'{ba} ~ {bio} + Age + Gender + education', data=subset_data).fit()
            
            # X, M → Y (biomarker, brain_age → cognition)
            outcome_model = ols(f'{cog} ~ {bio} + {ba} + Age + Gender + education', data=subset_data).fit()

            # 
            a_coeff = med_model.params[bio]  # X → M 
            b_coeff = outcome_model.params[ba]  # M → Y 

            indirect_effect = a_coeff * b_coeff  
            direct_effect = outcome_model.params[bio]
            total_effect = direct_effect + indirect_effect

            print(f"Indirect Effect: {indirect_effect:.3f}")
            print(f"Direct Effect: {direct_effect:.3f}")
            print(f"Total Effect: {total_effect:.3f}")

            # Bootstrap for confidence intervals
            bootstrap_estimates = Parallel(n_jobs=-1)(
                delayed(bootstrap_effect)(data.sample(frac=1, replace=True).dropna(subset=[bio, ba, cog]), bio, ba, cog)
                for _ in range(bootstrap_samples)
            )

            # 过滤掉失败的 bootstrap 结果
            bootstrap_estimates = [est for est in bootstrap_estimates if est is not None]
            
            if bootstrap_estimates:
                ci_lower = np.percentile(bootstrap_estimates, 100 * (alpha / 2))
                ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
                p_value = np.mean(np.array(bootstrap_estimates) < 0 if indirect_effect > 0 else np.array(bootstrap_estimates) > 0)

                print(f"95% CI for Indirect Effect: ({ci_lower:.3f}, {ci_upper:.3f})")
                print(f"p-value for Indirect Effect: {p_value:.4f}")
            else:
                ci_lower, ci_upper, p_value = np.nan, np.nan, np.nan
                print(f"Bootstrap failed for {bio} → {ba} → {cog}")

            # 结果存储
            results.append({
                "Biomarker": bio,
                "Brain Age": ba,
                "Cognition": cog,
                "X_to_M_effect": a_coeff.round(3), 
                "M_to_Y_effect": b_coeff.round(3),
                "Indirect Effect": indirect_effect.round(3),
                "Direct Effect": direct_effect.round(3),
                "Total Effect": total_effect.round(3),
                "CI Lower": ci_lower,
                "CI Upper": ci_upper,
                "P-Value": p_value
            })

results_df = pd.DataFrame(results)
print(results_df)
filtered_results_df = results_df[results_df["P-Value"] < 0.05]

output_dir = 'adni_BA1'
os.makedirs(output_dir, exist_ok=True)
filtered_results_df.to_csv(os.path.join(output_dir, 'mediation_results_significant_fu_fdr.csv'), index=False)

# _, pvals_corrected, _, _ = multipletests(results_df["P-Value"], method='bonferroni') #fdr_bh
_, pvals_corrected, _, _ = multipletests(results_df["P-Value"], method='fdr_bh')
results_df["P-Value Corrected"] = pvals_corrected
filtered_fdr_results_df = results_df[results_df["P-Value Corrected"] < 0.05]
filtered_fdr_results_df.to_csv(os.path.join(output_dir, 'mediation_results_significant_corrected_fu_fdr.csv'), index=False)

results_df.to_csv(os.path.join(output_dir, 'mediation_results_fu_fdr.csv'), index=False)
