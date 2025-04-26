'''
Machine learning model for predicting MCI-to-AD conversion
• Model 1 (benchmark): genetic and pathological biomarkers only;
• Model 2: Model 1plus baseline PADs;
• Model 3: Model 2 plus short-term longitudinal PAD change rates.

'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def calculate_annual_change(group, features):
    change_rates = {}

    for feature in features:
        if len(group) > 1:
            start_value = group.iloc[0][feature]
            end_value = group.iloc[-1][feature]
            start_time = group.iloc[0]['Time']
            end_time = group.iloc[-1]['Time']
            
            annual_change = (end_value - start_value) / (end_time - start_time)
            change_rates[feature] = annual_change
        else:
            change_rates[feature] = np.nan  
            
    return pd.Series(change_rates)

data = pd.read_csv('adni_BA/lme_data_3group_6yr_not_3sigma.csv') 
data = data[data['Group'].isin(['MCI_to_MCI', 'MCI_to_AD'])]

stat_data = data[data['Time'] != 0]
subject_time_counts = stat_data.groupby('Subject ID')['Time'].count()
subject_ids_to_keep = subject_time_counts[subject_time_counts >= 2].index
stat_data = stat_data[(stat_data['Subject ID'].isin(subject_ids_to_keep))]
print(stat_data.groupby('Group')['Subject ID'].nunique())

columns_to_calculate = ['VIS', 'SM', 'DAN', 'VAN', 'LIM', 'FP', 'DMN']
# 选取前两次检查的数据
stat_data = stat_data.sort_values(['Subject ID', 'Time'])
last_two_checks = stat_data.groupby('Subject ID').head(2)

print(last_two_checks.groupby('Time')['Subject ID'].nunique())

last_two_checks_sorted = last_two_checks.sort_values(by=['Subject ID', 'Time'])
last_two_checks_sorted['is_BL'] = last_two_checks_sorted.groupby('Subject ID')['Time'].transform('min') == last_two_checks_sorted['Time']
BL_data = last_two_checks_sorted[last_two_checks_sorted['is_BL']]
print(BL_data)  

change_rates_df = stat_data.groupby('Subject ID').apply(calculate_annual_change, features=columns_to_calculate)
all_individual_rates = stat_data[['Subject ID', 'Group']].drop_duplicates().merge(change_rates_df, on='Subject ID', how='left')
all_individual_rates = all_individual_rates.rename(columns={col: col + '_rate' for col in change_rates_df.columns})

pad_merged = pd.merge(BL_data, all_individual_rates,on=['Subject ID', 'Group'], how='inner')

biomarker_df = pd.read_csv('corr/bl_rate_corr_data.csv')[['Subject ID','PHS','ABETA42', 'TAU', 'PTAU', 'HCI', 'ENTORHINAL_SUVR', 'INFERIOR_TEMPORAL_SUVR','TAU_METAROI']]

merged_data = pd.merge(pad_merged,biomarker_df,on=['Subject ID'], how='inner')
print(merged_data)
print(merged_data.columns)

def compute_metrics(X, y, suffix, return_results=True):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y_encoded)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=39)

    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []
    accuracies, f1_scores = [], []
    fold_sensitivities, fold_specificities = [], []

    plt.figure(figsize=(6, 6))
    for i, (train_index, test_index) in enumerate(kf.split(X_resampled, y_resampled)):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        rf_model = RandomForestClassifier(n_estimators=100, random_state=40, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        y_pred = rf_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        accuracies.append(acc)
        f1_scores.append(f1)
        fold_sensitivities.append(sensitivity)
        fold_specificities.append(specificity)
        aucs.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        plt.plot(fpr, tpr, lw=3, alpha=0.6, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    output_dir = 'figure/predict'
    os.makedirs(output_dir, exist_ok=True)
    plt.plot(mean_fpr, mean_tpr, color='blue', lw=3, linestyle='--',
             label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance', lw=3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {suffix}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'roc_curve_{suffix}.pdf'), format='pdf', dpi=600)
    plt.close()

    if return_results:
        return {
            'suffix': suffix,
            'accuracy': accuracies,
            'auc': aucs,
            'f1': f1_scores,
            'sensitivity': fold_sensitivities,
            'specificity': fold_specificities
        }
    
from scipy.stats import ttest_rel, wilcoxon

def compare_models(result_a, result_b, metric='auc', method='ttest'):
    vals_a = result_a[metric]
    vals_b = result_b[metric]
    suffix_a = result_a['suffix']
    suffix_b = result_b['suffix']

    if method == 'ttest':
        stat, p = ttest_rel(vals_a, vals_b)
    elif method == 'wilcoxon':
        stat, p = wilcoxon(vals_a, vals_b)
    else:
        raise ValueError("method must be 'ttest' or 'wilcoxon'")

    print(f"Comparison [{suffix_b} vs {suffix_a}] on {metric.upper()}:")
    print(f"Mean {suffix_a}: {np.mean(vals_a):.3f}, Mean {suffix_b}: {np.mean(vals_b):.3f}")
    print(f"{method} t-value: {stat:.4f}\n p-value: {p:.4f}\n")

# feature selection
X1 = merged_data[columns_to_calculate]
X2 = merged_data[columns_to_calculate + ['VIS_rate', 'SM_rate', 'DAN_rate', 'VAN_rate', 'LIM_rate', 'FP_rate','DMN_rate']]
biomarker =  ['PHS','ABETA42', 'TAU', 'PTAU', 'HCI', 'ENTORHINAL_SUVR', 'INFERIOR_TEMPORAL_SUVR','TAU_METAROI']
X3 = merged_data[biomarker]
X4 = merged_data[biomarker + columns_to_calculate + ['VIS_rate', 'SM_rate', 'DAN_rate', 'VAN_rate', 'LIM_rate', 'FP_rate','DMN_rate']]

res_bio = compute_metrics(X3, merged_data['Group'], 'bio')
res_pad = compute_metrics(X1, merged_data['Group'], 'pad_bl')
res_bio_pad = compute_metrics(X4, merged_data['Group'], 'bio_pad_bl_fu')

compare_models(res_bio, res_pad, metric='auc')
compare_models(res_pad, res_bio_pad, metric='auc')
