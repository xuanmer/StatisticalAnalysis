'''
A generalized linear model was utilized separately to examine the associations between baseline-PADs and PAD change rates with these measurements within MCI.

Group
MCI_to_MCI    300
MCI_to_AD     157
'''

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats


data = pd.read_csv('corr/bl_rate_corr_data.csv') 

cognitive = ['MMSCORE', 'MOCA']
genetic = ['PHS']
biomarker = ['ABETA42', 'TAU', 'PTAU', 'HCI', 'ENTORHINAL_SUVR', 'INFERIOR_TEMPORAL_SUVR','TAU_METAROI'] # 'CENTILOIDS', 
mental_health = ['GDTOTAL','NPISCORE']
features = genetic + ['APOE4'] + biomarker + cognitive + mental_health

print(data['Group'].value_counts())
data['Group'] = pd.Categorical(data['Group'], categories=['MCI_to_MCI', 'MCI_to_AD'], ordered=True)

results_list = []
final_results = pd.DataFrame({"Feature": features}, columns=['Feature'])

networks = ['VIS', 'SM', 'DAN', 'VAN', 'LIM', 'FP', 'DMN', 'VIS_PAD_rate', 'SM_PAD_rate', 'DAN_PAD_rate', 'VAN_PAD_rate', 'LIM_PAD_rate', 'FP_PAD_rate', 'DMN_PAD_rate']

scaler = StandardScaler()
data[features+networks] = scaler.fit_transform(data[features+networks])
for xindex in networks:
    array_r = []
    array_p = []

    for yindex in features:
        if yindex == 'APOE4':
            stat_data = data.loc[:, [xindex, "Group", "Age", "education", "Gender", yindex]].dropna()
            formula = f'{yindex} ~ {xindex} + Age + education + Gender'
        else:
            stat_data = data.loc[:, [xindex, "Group", "Age", "education", "Gender", "APOE4", yindex]].dropna()
            formula = f'{yindex} ~ {xindex} + Age + education + Gender + APOE4'

        glm_model = sm.formula.glm(formula=formula, data=stat_data, family=sm.families.Gaussian())
        result = glm_model.fit()
        print(result.summary())

        array_r.append(result.params.get(f"{xindex}", None))
        array_p.append(result.pvalues.get(f"{xindex}", None))
    
    fdr_corrected_p = pg.multicomp(array_p, method='fdr_bh')[1]

    temp_df = pd.DataFrame({
        "Feature": features,
        f"{xindex}-r": array_r,
        f"{xindex}-p": array_p,
        f"{xindex}-fdr_bh": fdr_corrected_p,
        # f"{xindex}-Interaction_coeff": interaction_coeff,
        # f"{xindex}-Interaction_p_value": interaction_p_value,
        # f"{xindex}-Interaction_fdr_bh": fdr_corrected_interaction_p
    })
    
    results_list.append(temp_df)
    final_results = pd.merge(final_results, temp_df, on='Feature', how='outer')

print(final_results)
final_results.to_csv('results/corr/corr_glm_results.csv', index=False)

min_r = final_results[[f"{xindex}-r" for xindex in networks]].min().min()
max_r = final_results[[f"{xindex}-r" for xindex in networks]].max().max()
output_dir = 'results/corr/bar_plots'
os.makedirs(output_dir, exist_ok=True) 

for xindex in networks:
    coefficients = final_results[f"{xindex}-r"]
    features = final_results['Feature']

    temp_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients
    }).dropna() 
    
    # temp_df = temp_df.sort_values(by="Coefficient", ascending=True)

    # plt.figure(figsize=(8, len(temp_df) * 0.5))  
    # sns.barplot(
    #     data=temp_df,
    #     x="Coefficient",
    #     y="Feature",
    #     palette="coolwarm",
    #     orient="h"
    # )
    # plt.xlim(min_r - 0.1, max_r + 0.1) 
    # plt.axvline(0, color='black', linestyle='--', linewidth=0.8)  # 
    # plt.title(f"Regression Coefficients for {xindex}", fontsize=16)
    # plt.xlabel("Coefficient (r)", fontsize=12)
    # plt.ylabel("Feature", fontsize=12)
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/{xindex}_coefficients_barplot.pdf", dpi=600)
    # plt.close()

# ----------------------------------------------------------------
heatmap_data = final_results.set_index('Feature')[[
    f"{xindex}-r" for xindex in networks
]].dropna()

plt.figure(figsize=(10, len(heatmap_data) * 0.5))
sns.heatmap(
    heatmap_data,
    annot=False,  # 
    cmap="coolwarm",  # 
    vmin=-1,vmax=1,  
    center=0,  
    cbar_kws={'label': 'Regression Coefficient (r)'}
)
plt.title("Regression Coefficients for Networks and Features", fontsize=16)
plt.xlabel("Networks", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.tight_layout()

output_path = "corr/heatmap_coefficients.pdf"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=600)


pd_array = pd.DataFrame(None)
for xindex in ['DMN','DAN_PAD_rate',]:
    print('-------------------')
    print(xindex)
    array_r = []
    array_p = []
    array_p_corrected = []
    for yindex in ['TAU']:
        stat_data = data.loc[:, [xindex,"Group", "Age", "education", "Gender","APOE4", yindex]].dropna()
        print(f'{xindex=}')
        print(f'{yindex=}')

        formula = f'{yindex} ~ {xindex} + Age + education + Gender + APOE4'
        glm_model = sm.formula.glm(formula=formula, data=stat_data, family=sm.families.Gaussian())
        result = glm_model.fit()

        plt.rcParams.update({'font.size': 22})
        stat_data['predicted'] = result.fittedvalues

        sns.lmplot(data=stat_data, 
           x=xindex,  # 
           y='predicted',  # 
           markers=['o'], 
           scatter_kws={'s': 50, 'color': '#B63E3A'},  # 
           line_kws={'color': '#B63E3A', 'linewidth': 3},  # 
           ci=95)  
        corr = result.params.get(f"{xindex}", None)
        p_value = result.pvalues.get(f"{xindex}", None)

        # 皮尔逊相关
        # X = stat_data[['Gender', 'Age', 'education',"APOE4"]]
        # y = stat_data[yindex]
        # sns.lmplot(data=stat_data, 
        #    x=xindex, 
        #    y='{}'.format(yindex), 
        #    markers=['o'], 
        #    scatter_kws={'s': 50, 'color': '#B63E3A'},  # 设置散点颜色
        #    line_kws={'color': '#B63E3A', 'linewidth': 3},  # 设置回归线颜色
        #    ci=95,
        #    palette='#B63E3A',  
        #    robust=True)  # #['#4F60CB','#B63E3A']
        
        # corr, p_value = stats.pearsonr(stat_data[xindex], stat_data['{}'.format(yindex)])
        # print(f'r = {corr:.2f} , p = {p_value:.3f}')

        # _, pvals_corrected, _, _ = multitest.multipletests(p_value, alpha=0.05, method='fdr_bh')
        # array_p_corrected.append(pvals_corrected)
        
        # 添加相关系数和 p 值
        plt.text(min(data[xindex]), max(data['{}'.format(yindex)]), f'r = {corr:.1f} , p = {p_value:.1e}', fontsize=12)
        # plt.show()
        plt.savefig('./figure/corr/{}_{}.pdf'.format(xindex, yindex), format='pdf', dpi=600)
        # plt.close()