
"""FSHD Module 2: interpolation + charts 1 and 2 (lecture-inspired percent-change comparisons).
Run beside Module2_FSHD_Dataset.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path.cwd()
CSV = next((p for p in (BASE/'Module 2/Module2_FSHD_Dataset.csv', BASE/'Module2_FSHD_Dataset(1).csv') if p.exists()), None)
if CSV is None:
    raise FileNotFoundError('Put Module2_FSHD_Dataset.csv in the same folder as this script/notebook.')

df = pd.read_csv(CSV)
T1 = float(df['lag_months_y1'].min())
T2 = float(df['lag_months_y2'].min())
print(f'Common visit times: T1={T1:.2f} months, T2={T2:.2f} months')

measures = {
    'fat_fraction': {'R':['ff_baseline_R','ff_y1_R','ff_y2_R'], 'L':['ff_baseline_L','ff_y1_L','ff_y2_L']},
    'dorsiflexion_strength': {'R':['dorsiflexion_qmt_R_baseline','dorsiflexion_qmt_R_y1','dorsiflexion_qmt_R_y2'],
                             'L':['dorsiflexion_qmt_L_baseline','dorsiflexion_qmt_L_y1','dorsiflexion_qmt_L_y2']}
}
labels = {'fat_fraction':'MRI fat fraction', 'dorsiflexion_strength':'Dorsiflexion strength'}
colors = {'fat_fraction':'#4C78A8', 'dorsiflexion_strength':'#F28E2B'}

def linear_interpolate(t, values, target):
    t, values = np.asarray(t, dtype=float), np.asarray(values, dtype=float)
    good = np.isfinite(t) & np.isfinite(values)
    t, values = t[good], values[good]
    if len(t) != 2 or t[0] >= t[1] or not (t[0] <= target <= t[1]):
        return np.nan
    return float(np.interp(target, t, values))

def quadratic_interpolate(t, values, target):
    t, values = np.asarray(t, dtype=float), np.asarray(values, dtype=float)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(values))):
        return np.nan
    if not (t[0] < t[1] < t[2] and t[0] <= target <= t[2]):
        return np.nan
    return float(np.polyval(np.polyfit(t, values, 2), target))

records=[]
for _, sub in df.iterrows():
    times = [sub['lag_months_baseline'], sub['lag_months_y1'], sub['lag_months_y2']]
    for measure, sides in measures.items():
        for side, cols in sides.items():
            y = [sub[c] for c in cols]
            v1 = linear_interpolate(times[:2], y[:2], T1)
            v2 = quadratic_interpolate(times, y, T2)
            records.append({'subject_id':sub['subject_id'], 'side':side, 'measure':measure,
                            't1_linear':v1, 't2_quadratic':v2})
results = pd.DataFrame(records)

wide = results.pivot(index=['subject_id','side'], columns='measure', values=['t1_linear','t2_quadratic'])
wide.columns = ['_'.join(c) for c in wide.columns]
paired = wide.dropna().reset_index()
subject = paired.groupby('subject_id', as_index=False).mean(numeric_only=True)
for measure in measures:
    subject[f'{measure}_T1'] = subject[f't1_linear_{measure}']
    subject[f'{measure}_T2'] = subject[f't2_quadratic_{measure}']
    subject[f'{measure}_absolute_change'] = subject[f'{measure}_T2'] - subject[f'{measure}_T1']
    subject[f'{measure}_percent_change'] = np.where(
        subject[f'{measure}_T1'].abs() > 1e-10,
        100 * subject[f'{measure}_absolute_change'] / subject[f'{measure}_T1'],
        np.nan
    )

print(f'Participants with both outcomes for paired analysis: {len(subject)} / {df.subject_id.nunique()}')

stat_rows=[]
for measure in ['fat_fraction','dorsiflexion_strength']:
    t1 = subject[f'{measure}_T1'].to_numpy(dtype=float)
    t2 = subject[f'{measure}_T2'].to_numpy(dtype=float)
    pct = subject[f'{measure}_percent_change'].to_numpy(dtype=float)
    ok = np.isfinite(t1) & np.isfinite(t2) & np.isfinite(pct)
    t1, t2, pct = t1[ok], t2[ok], pct[ok]
    delta = t2 - t1
    t_stat, p_val = stats.ttest_rel(t2, t1) if len(delta) >= 2 else (np.nan, np.nan)
    stat_rows.append({
        'measure': measure,
        'label': labels[measure],
        'n': len(delta),
        'mean_pct': np.mean(pct),
        'sd_pct': np.std(pct, ddof=1),
        'mean_over_sd': np.mean(pct) / np.std(pct, ddof=1) if np.std(pct, ddof=1) > 0 else np.nan,
        'p_value': p_val,
        'T1_mean': np.mean(t1),
        'T2_mean': np.mean(t2)
    })
stat_table = pd.DataFrame(stat_rows)
print(stat_table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# ----- Figure 1: lecture-style percent change summary -----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
all_vals = []
for measure in ['fat_fraction', 'dorsiflexion_strength']:
    vals = subject[f'{measure}_percent_change'].dropna().to_numpy(dtype=float)
    all_vals.extend(vals.tolist())
ymin = min(-30, np.floor(min(all_vals)/5)*5 - 5)
ymax = max(30, np.ceil(max(all_vals)/5)*5 + 10)

for ax, measure in zip(axes, ['fat_fraction', 'dorsiflexion_strength']):
    vals = subject[f'{measure}_percent_change'].dropna().to_numpy(dtype=float)
    vals = np.sort(vals)[::-1]
    x = np.arange(1, len(vals)+1)
    ax.bar(x, vals, color=colors[measure], alpha=0.85, width=0.75)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(np.mean(vals), color='black', linestyle='--', linewidth=1.5)
    row = stat_table.loc[stat_table['measure'] == measure].iloc[0]
    signal_text = 'Realer signal' if abs(row['mean_over_sd']) >= 0.3 else 'Mostly noise'
    marker = '✓' if abs(row['mean_over_sd']) >= 0.3 else '✗'
    ax.set_title(f"{labels[measure]}\n{marker} {signal_text}", fontsize=13)
    ax.set_xlabel('Participant (sorted by % change)')
    ax.set_ylim(ymin, ymax)
    summary_text = (
        f"% change, T1 → T2\n"
        f"mean = {row['mean_pct']:.2f}%\n"
        f"SD = {row['sd_pct']:.2f}%\n"
        f"mean/SD = {row['mean_over_sd']:.2f}\n"
        f"paired t-test p = {row['p_value']:.3g}\n"
        f"n = {int(row['n'])}"
    )
    ax.text(0.03, 0.97, summary_text, transform=ax.transAxes, va='top', ha='left',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='gray', alpha=0.95))
axes[0].set_ylabel('Relative change from T1 to T2 (%)')
fig.suptitle('Figure 1. Percent change comparison (lecture-style summary)', fontsize=14)
fig.tight_layout()
plt.show()

# ----- Figure 2: individual percent changes -----
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, measure in enumerate(['fat_fraction', 'dorsiflexion_strength']):
    vals = subject[f'{measure}_percent_change'].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    jitter = np.random.default_rng(2315+i).uniform(-.13, .13, len(vals))
    ax.scatter(i + jitter, vals, alpha=.78, s=40, color=colors[measure], edgecolors='white', linewidths=.4)
    ax.hlines(np.mean(vals), i-.24, i+.24, color='black', linewidth=3, label='Mean' if i == 0 else None)
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xticks([0, 1], [labels['fat_fraction'], labels['dorsiflexion_strength']])
ax.set_ylabel('Individual T1 to T2 relative change (%)')
ax.set_title('Figure 2. Participant-level variation in progression measures')
ax.legend()
fig.tight_layout()
plt.show()
