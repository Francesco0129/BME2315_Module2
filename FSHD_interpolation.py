"""FSHD Module 2: subject-level time interpolation. Run beside Module2_FSHD_Dataset.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
CSV = next((p for p in (BASE/'Module2_FSHD_Dataset.csv', BASE/'Module2_FSHD_Dataset(1).csv') if p.exists()), None)
if CSV is None:
    raise FileNotFoundError('Put Module2_FSHD_Dataset.csv in the same folder as this script/notebook.')
df = pd.read_csv(CSV)
T1 = float(df['lag_months_y1'].min())
T2 = float(df['lag_months_y2'].min())
print(f'Common visit times: T1={T1:.2f} months, T2={T2:.2f} months')

# Use one row per subject AND side. Two sides of the same participant are
# NOT independent observations for hypothesis testing; that comes later.
measures = {
    'fat_fraction': {'R':['ff_baseline_R','ff_y1_R','ff_y2_R'], 'L':['ff_baseline_L','ff_y1_L','ff_y2_L']},
    'dorsiflexion_strength': {'R':['dorsiflexion_qmt_R_baseline','dorsiflexion_qmt_R_y1','dorsiflexion_qmt_R_y2'],
                             'L':['dorsiflexion_qmt_L_baseline','dorsiflexion_qmt_L_y1','dorsiflexion_qmt_L_y2']}
}

def linear_interpolate(t, values, target):
    """Linear interpolation on two real, finite bracketing measurements."""
    t, values = np.asarray(t, dtype=float), np.asarray(values, dtype=float)
    good = np.isfinite(t) & np.isfinite(values)
    t, values = t[good], values[good]
    if len(t) != 2 or t[0] >= t[1] or not(t[0] <= target <= t[1]):
        return np.nan
    return float(np.interp(target, t, values))

def quadratic_interpolate(t, values, target):
    """Degree-2 interpolating polynomial through all 3 real visits (no extrapolation)."""
    t, values = np.asarray(t, dtype=float), np.asarray(values, dtype=float)
    if not(np.all(np.isfinite(t)) and np.all(np.isfinite(values))):
        return np.nan
    if not(t[0] < t[1] < t[2] and t[0] <= target <= t[2]):
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
            # Linear-at-T2 retained as a sensitivity check (optional).
            v2_linear = linear_interpolate(times[1:], y[1:], T2)
            records.append({'subject_id':sub['subject_id'], 'side':side,
                            'measure':measure, 'baseline':y[0], 'year1_actual':y[1],
                            'year2_actual':y[2], 't1_linear':v1,
                            't2_quadratic':v2, 't2_linear_sensitivity':v2_linear,
                            'change_t1_to_t2':v2-v1 if np.isfinite(v1) and np.isfinite(v2) else np.nan})
results = pd.DataFrame(records)
results.to_csv(BASE/'FSHD_interpolated_all_sides.csv', index=False)
print('\nAvailable interpolation values (out of 34 subjects per side):')
print(results.groupby(['measure','side'])[['t1_linear','t2_quadratic','change_t1_to_t2']].count().to_string())

# For a genuinely subject-level comparison, average left and right only if BOTH
# measures are available in a given side at both targets; pairing avoids unequal cohorts.
wide = results.pivot(index=['subject_id','side'], columns='measure', values=['t1_linear','t2_quadratic'])
wide.columns=['_'.join(c) for c in wide.columns]
paired = wide.dropna().reset_index()
subject = paired.groupby('subject_id',as_index=False).mean(numeric_only=True)
for measure in measures:
    subject[f'{measure}_T1'] = subject[f't1_linear_{measure}']
    subject[f'{measure}_T2'] = subject[f't2_quadratic_{measure}']
    subject[f'{measure}_absolute_change'] = subject[f'{measure}_T2']-subject[f'{measure}_T1']
    subject[f'{measure}_percent_change'] = np.where(
        subject[f'{measure}_T1'].abs()>1e-10,
        100*subject[f'{measure}_absolute_change']/subject[f'{measure}_T1'], np.nan)
subject.to_csv(BASE/'FSHD_interpolated_subjects_paired.csv',index=False)
print(f'\nParticipants with both outcomes for paired analysis: {len(subject)} / {df.subject_id.nunique()}')
for measure in measures:
    x = subject[f'{measure}_percent_change'].dropna()
    print(f'{measure}: mean percent change={x.mean():.2f}%, SD={x.std(ddof=1):.2f}%, N={len(x)}')

# Validation chart: one person, both visits, actual data and fitted curves.
# Generated with matplotlib; saved separately for pasting into the main notebook.
from matplotlib.lines import Line2D
valid_ids = subject['subject_id'].tolist()
if valid_ids:
    pid=valid_ids[0]
    original=df.loc[df.subject_id==pid].iloc[0]
    xx=np.array([original['lag_months_baseline'],original['lag_months_y1'],original['lag_months_y2']],dtype=float)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,(measure, cols) in zip(axes,[(m,measures[m]['R']) for m in measures]):
        yy=np.array([original[c] for c in cols],dtype=float)
        ax.scatter(xx,yy,s=50,color='black',label='Actual visits',zorder=4)
        dense=np.linspace(xx[0],xx[-1],150)
        ax.plot(dense[dense<=xx[1]],np.interp(dense[dense<=xx[1]],xx[:2],yy[:2]),label='Linear: baseline to Y1')
        ax.plot(dense,np.polyval(np.polyfit(xx,yy,2),dense),label='Quadratic: all 3 visits',linestyle='--')
        ax.axvline(T1,color='gray',alpha=.55);ax.axvline(T2,color='gray',alpha=.55)
        ax.set(xlabel='Months after baseline',ylabel='Fat fraction (%)' if measure=='fat_fraction' else 'Dorsiflexion QMT (dataset units)',title=f'{measure.replace("_"," ").title()} (right side)')
        ax.legend(fontsize=8)
    fig.suptitle(f'Interpolation validation — subject {pid}')
    fig.tight_layout()
    fig.savefig(BASE/'FSHD_interpolation_validation.png',dpi=170,bbox_inches='tight')
    plt.show()
