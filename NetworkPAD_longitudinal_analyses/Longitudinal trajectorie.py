import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2
from sklearn.model_selection import train_test_split

# step1.  按诊断转换分组
def has_recurrent_diagnosis(diagnoses):
    seen = set()
    last_state = None
    for diag in diagnoses:
        if diag != last_state:
            if diag in seen:
                return True
            seen.add(diag)
            last_state = diag
    return False

def has_invalid_diagnosis_transition(diagnoses):
    valid_order = {"CN": 0, "MCI": 1, "AD": 2}
    compressed_diagnoses = [diagnoses[i] for i in range(len(diagnoses)) if i == 0 or diagnoses[i] != diagnoses[i - 1]]
    numeric_diagnoses = [valid_order[diag] for diag in compressed_diagnoses]
    for i in range(1, len(numeric_diagnoses)):
        if numeric_diagnoses[i] < numeric_diagnoses[i - 1]:
            return True
    return False

def classify_diagnosis_transitions(group):
    diagnoses = group.sort_values('Age')['DIAGNOSIS'].tolist()
    unique_diagnoses = set(diagnoses)

    is_recurrent = has_recurrent_diagnosis(diagnoses)
    is_invalid = has_invalid_diagnosis_transition(diagnoses)

    if is_recurrent:
        return {'CN_to_AD': False, 'CN_to_MCI': False, 'CN_to_CN': False,
                'MCI_to_MCI': False, 'MCI_to_AD': False, 'AD_to_AD': False,
                'Recurrent': True, 'Invalid': False}
    if is_invalid:
        return {'CN_to_AD': False, 'CN_to_MCI': False, 'CN_to_CN': False,
                'MCI_to_MCI': False, 'MCI_to_AD': False, 'AD_to_AD': False,
                'Recurrent': False, 'Invalid': True}

    transitions = {
        'CN_to_AD': 'CN' in diagnoses and 'AD' in diagnoses,
        'CN_to_MCI': 'CN' in diagnoses and 'MCI' in diagnoses and 'AD' not in diagnoses,
        'CN_to_CN': unique_diagnoses == {'CN'},
        'MCI_to_MCI': unique_diagnoses == {'MCI'},
        'MCI_to_AD': 'MCI' in diagnoses and 'AD' in diagnoses,
        'AD_to_AD': unique_diagnoses == {'AD'},
        'Recurrent': False,
        'Invalid': False
    }
    return transitions

# 输入数据: predicted age difference
data = pd.read_csv('adni_BA/merged_pad.csv')

transition_results = (
    data.groupby('Subject ID')
    .apply(classify_diagnosis_transitions)
    .reset_index(name='Transitions')
)

def get_group_label(transitions):
    # 找到 transitions 中为 True 的键，作为 group 名
    for key, value in transitions.items():
        if value:
            return key
    return 'Unknown'

transition_results['Group'] = transition_results['Transitions'].apply(get_group_label)
data = pd.merge(data, transition_results[['Subject ID', 'Group']], on='Subject ID')

print('================================================================')
# 按 Group 统计唯一 Subject ID 的数量
data = data[data['Group'] != 'CN_to_AD']
group_counts = data.groupby('Group')['Subject ID'].nunique()
print(group_counts)

# Step 2_1_2: 'CN_to_AD' 或 'MCI_to_AD', 'MCI_to_AD'，转化组 Time = 0 : 为首次诊断时的年龄; 对于其他 非转化组（如 'CN_to_CN'、'MCI_to_MCI'、'AD_to_AD'），0年 表示随访结束的年份。
stat_data = data[~data['Group'].isin(['Recurrent', 'Invalid'])].copy()
max_age = stat_data.groupby('Subject ID')['Age'].transform('max')

def calculate_baseline(row):
    if row['Group'] in ['CN_to_AD', 'MCI_to_AD'] and row['DIAGNOSIS'] == 'AD':
        return row['Age']  
    elif row['Group'] in ['CN_to_MCI'] and row['DIAGNOSIS'] == 'MCI':
        return row['Age']  
    else:
        return max_age[row.name]  

stat_data['Baseline_Age'] = stat_data.apply(calculate_baseline, axis=1)
stat_data['Baseline_Age'] = stat_data.groupby('Subject ID')['Baseline_Age'].transform('min')
stat_data['Time'] = stat_data['Age'] - stat_data['Baseline_Age']
stat_data['Time'] = stat_data['Time'].round().astype(int)
print(stat_data[['Subject ID', 'Age', 'DIAGNOSIS', 'Group', 'Baseline_Age', 'Time']])
stat_data = stat_data[(stat_data['Time'] <= 0) & (stat_data['Time'] >= -6)] 

# 不保留只有一个时间点的记录，保证至少有一次随访记录
subject_counts = stat_data['Subject ID'].value_counts()
valid_subjects = subject_counts[subject_counts > 1].index
stat_data = stat_data[stat_data['Subject ID'].isin(valid_subjects)]

# 保留 有 Time 次数 >=3 的记录
subject_time_counts = stat_data.groupby('Subject ID')['Time'].count()
subject_ids_to_keep = subject_time_counts[subject_time_counts >= 3].index
stat_data = stat_data[
    (stat_data['Group'].isin(['CN_to_CN']) & stat_data['Subject ID'].isin(subject_ids_to_keep)) |
    (~stat_data['Group'].isin(['CN_to_CN']))
]

group_counts = stat_data.groupby('Group')['Subject ID'].nunique()
print(group_counts)

features = ['VIS', 'SM', 'DAN', 'VAN', 'LIM', 'FP', 'DMN']
column_mapping = {f'Net{i}_Corrected_PAD': name for i, name in enumerate(features, start=1)}
stat_data = stat_data.rename(columns=column_mapping)

cols = ['Subject ID', 'Age', 'DIAGNOSIS', 'Group', 'Time', 'Gender', 'education', 'APOE4','imageId'] + features
stat_data = stat_data[cols]
# print(stat_data)
stat_data = stat_data[stat_data['Group'].isin(['CN_to_CN', 'MCI_to_MCI', 'MCI_to_AD'])]
print(stat_data.groupby('Group')['Subject ID'].nunique())

# ----------------------------------------------------------------
all_individual_rates = stat_data[['Subject ID', 'Group']] # 记录每个个体的变化率
merge_all_predictions = pd.DataFrame()

for net in features:
    df = stat_data.dropna(subset=['Subject ID', 'Time', 'Group', 'Age', 'DIAGNOSIS', 'Gender', 'education', 'APOE4', net])

    # grouped = df.groupby('Group')
    # filtered_data = []
    # for group_name, group_data in grouped:

    #     mean = group_data[net].mean()
    #     std_dev = group_data[net].std()
    #     lower_bound = mean - 3 * std_dev
    #     upper_bound = mean + 3 * std_dev

    #     filtered_group = group_data[(group_data[net] >= lower_bound) & (group_data[net] <= upper_bound)]
    #     filtered_data.append(filtered_group)

    # filtered_df = pd.concat(filtered_data)
    
    # stat_data = stat_data[~stat_data.index.isin(df.index)]  # 移除当前特征对应的数据
    # stat_data = pd.concat([stat_data, filtered_df])  # 加入筛选后的数据

    model = MixedLM.from_formula(
        f'{net} ~ Time + Group + Age + Gender + education + APOE4 + Group: Time', 
        groups='Subject ID', 
        re_formula='~ Time',
        data=df)
    result = model.fit(method='nm', maxiter=1000)
    print(result.summary())

    df[f'Predicted_{net}'] = result.predict(df)
    all_predictions = df[df['Group'] == 'MCI_to_AD'][['Time', f'Predicted_{net}']]
    all_predictions['Feature'] = net
    all_predictions = all_predictions.rename(columns={f'Predicted_{net}': 'Predicted'})
    merge_all_predictions = pd.concat([merge_all_predictions, all_predictions])

    # 固定效应的 Time 系数
    time_coef = result.fe_params['Time']  
    # Group:Time 交互项系数
    group_time_coefs = {}
    for group in df['Group'].unique():
        interaction_term = f'Group[T.{group}]:Time'
        group_time_coefs[group] = result.fe_params[interaction_term] if interaction_term in result.fe_params.index else 0
    group_rate = {group: time_coef + group_time_coefs[group] for group in df['Group'].unique()}
    group_rate['Net'] = net

    # 每个 Subject ID 的随机效应
    random_effects = result.random_effects

    individual_rates = pd.DataFrame()

    for subject_id, random_effect in random_effects.items():
        group = df[df['Subject ID'] == subject_id]['Group'].iloc[0]  
        individual_rate = {
            'Subject ID': subject_id,
            'Group': group,
            f'{net}_PAD_rate': time_coef + group_time_coefs[group] + random_effect['Time'] 
        }
        individual_rates = pd.concat([individual_rates, pd.DataFrame([individual_rate])])
    all_individual_rates = all_individual_rates.merge(individual_rates, on=['Subject ID','Group'], how='left')

    df[f'Predicted_{net}'] = result.predict(df)
    plt.rcParams.update({'font.size': 18})

    custom_palette = {
        'AD_to_AD': '#876096',
        'MCI_to_AD': '#A64C78',
        'MCI_to_MCI': '#4c7ba6',
        'CN_to_MCI': '#4c7ba6',
        'CN_to_CN': 'green'
    }
    custom_linestyles = {
        'AD_to_AD': 'solid',
        'MCI_to_AD': 'dashed',
        'MCI_to_MCI': 'solid',
        'CN_to_MCI': 'dashed',
        'CN_to_CN': 'solid'
    }
    g = sns.FacetGrid(df, hue="Group", palette=custom_palette, height=4, aspect=1.4)
    for group, linestyle in custom_linestyles.items():
        subset = df[df['Group'] == group]
        sns.regplot(
            data=subset,
            x="Time",
            y=f"Predicted_{net}",
            scatter=False,
            order=2, 
            ax=g.ax,
            label=group,
            line_kws={"linestyle": linestyle, "color": custom_palette[group], "linewidth": 3}
        )

    g.ax.legend(
    title="Group",
    fontsize=12,
    title_fontsize=14,
    loc="center left", 
    bbox_to_anchor=(1.0, 0.5) 
    )
    g.ax.set_xlabel('Time', fontsize=18)
    g.ax.set_ylabel(f'{net} PAD', fontsize=18)
    plt.tight_layout()
    # plt.savefig(f'figure/net_pad_trajectory/{net}_6yr.pdf')
    plt.cla(); plt.clf(); plt.close()

custom_palette = ['#006657','#e2ab75','#d19792','#8ca5c0','#ee765e','#053154', '#A42423']
plt.figure(figsize=(6, 6))
sns.set_theme(style="white")
sns.lmplot(data=merge_all_predictions, x="Time", y="Predicted", hue="Feature", aspect=1.1, height=6, ci=None, order=1, palette=custom_palette,scatter=False, legend=False,line_kws={"linewidth": 3.5})
plt.xlabel("Years before Diagnosis", fontsize=20)
plt.ylabel("PAD", fontsize=18)
plt.legend(title="Network", fontsize=12, title_fontsize=14, loc="lower right")
plt.grid(alpha=0.3)
plt.xticks(fontsize=24) 
plt.yticks(fontsize=20) 
plt.tight_layout()

plt.xlim(-6, 0)
plt.savefig(f'figure/net_pad_trajectory/net_pad_trajector_6yr.pdf')

all_individual_rates = all_individual_rates.drop_duplicates()
print(all_individual_rates)
all_individual_rates.round(3).to_csv('./individual_pad_rate.csv', index=False)
