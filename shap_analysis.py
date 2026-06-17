"""
shap_analysis.py
────────────────────────────────────────────────────────────────
SHAP analysis for the Random Forest TIDE 2.0 pipeline.
Run AFTER rf_pipeline.py (requires rf_tide_model.pkl).

Produces:
  shap_summary_bar.png        – mean |SHAP| bar chart
  shap_summary_beeswarm.png   – beeswarm (value + direction)
  shap_dependence_grid.png    – top-4 feature dependence plots
  shap_waterfall_examples.png – waterfall for one TP and one FP
  shap_values.csv             – raw SHAP values for all test rows
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

# ── Colour palette (matches rf_pipeline.py) ──────────────────────────────────
DARK_BG  = '#0f1117'
CARD_BG  = '#1e2130'
ACCENT1  = '#4C9BE8'
ACCENT2  = '#F06292'
ACCENT3  = '#81C784'
TEXT_COL = '#E0E0E0'
GRID_COL = '#2a2d3a'

plt.rcParams.update({
    'text.color':        TEXT_COL,
    'axes.labelcolor':   TEXT_COL,
    'xtick.color':       TEXT_COL,
    'ytick.color':       TEXT_COL,
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    CARD_BG,
    'axes.edgecolor':    GRID_COL,
    'grid.color':        GRID_COL,
    'font.family':       'DejaVu Sans',
})

# ─────────────────────────────────────────────
# 1. RE-BUILD DATA  (same logic as rf_pipeline.py)
# ─────────────────────────────────────────────
df = pd.read_excel('Tide_2_0_final.xlsx')

def clean_target(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if s in ('yes', 'y'): return 1
    if s in ('no', 'n', '-', '0'): return 0
    keywords = ['funding','grant','investment','investor','crore','lakh','received']
    if any(k in s for k in keywords): return 1
    return np.nan

def parse_amount(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower().replace(',','').replace('₹','').replace('rs','').replace('inr','').strip()
    m = re.match(r'([\d.]+)\s*(lakh|lac|l)', s)
    if m: return float(m.group(1)) * 100_000
    m = re.match(r'([\d.]+)\s*(crore|cr)', s)
    if m: return float(m.group(1)) * 10_000_000
    try: return float(s)
    except: return np.nan

def parse_team(val):
    if pd.isna(val): return np.nan
    try: return float(str(val).strip().split()[0])
    except: return np.nan

def parse_trl(val):
    if pd.isna(val): return np.nan
    m = re.search(r'\d+', str(val))
    if m:
        v = float(m.group())
        return v if 1 <= v <= 9 else np.nan
    return np.nan

def clean_tier(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    if '1' in s or s.endswith('I'):  return 'I'
    if '2' in s or 'II' in s:        return 'II'
    if '3' in s or 'III' in s:       return 'III'
    return np.nan

stage_map = {
    'ideation':'Ideation','proof of concept':'POC','poc':'POC',
    'prototype':'Prototype','prototype development':'Prototype',
    'mvp':'MVP','validation':'Validation','pilot':'Pilot',
    'pre-revenue':'Pre-Revenue','pre revenue':'Pre-Revenue','revenue':'Revenue',
}
def clean_stage(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    for k, v in stage_map.items():
        if k in s: return v
    return 'Other'

def clean_hwsw(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    if 'both' in s:  return 'Both'
    if 'hard' in s:  return 'Hardware'
    if 'soft' in s:  return 'Software'
    return 'Other'

def clean_women(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    return 'Yes' if s == 'yes' else 'No'

df['target']            = df['Whether Follow-on Funding Received'].apply(clean_target)
df['amount_sanctioned'] = df['Amount sanctioned under the scheme'].apply(parse_amount)
df['team_size']         = df['Team size at the time of onboarding'].apply(parse_team).clip(0, 100)
df['trl']               = df['Technology Readiness Level (TRL, at the time of onboarding)'].apply(parse_trl)
df['tier']              = df['Tier I / II / III'].apply(clean_tier)
df['stage']             = df['Stage of the Startup at the time of onboarding'].apply(clean_stage)
df['hw_sw']             = df['Is it a hardware startup or software startup'].apply(clean_hwsw)
df['women_founder']     = df['Is there a Women Founder/Co-Founder'].apply(clean_women)
sector_counts           = df['Sector'].value_counts()
top_sectors             = sector_counts[sector_counts >= 20].index
df['sector']            = df['Sector'].apply(lambda x: x if x in top_sectors else 'Other')

NUMERIC_FEATS = ['amount_sanctioned', 'team_size', 'trl']
CATEG_FEATS   = ['sector', 'tier', 'stage', 'hw_sw', 'women_founder']
ALL_FEATS     = NUMERIC_FEATS + CATEG_FEATS

# Human-readable feature labels
FEAT_LABELS = {
    'amount_sanctioned': 'Amount Sanctioned',
    'team_size':         'Team Size',
    'trl':               'TRL',
    'sector':            'Sector',
    'tier':              'Tier',
    'stage':             'Stage',
    'hw_sw':             'HW / SW',
    'women_founder':     'Women Founder',
}

model_df = df[ALL_FEATS + ['target']].dropna(subset=['target'])
X = model_df[ALL_FEATS]
y = model_df['target'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 2. LOAD MODEL & EXTRACT RF STEP
# ─────────────────────────────────────────────
print("Loading model …")
pipeline = joblib.load('rf_tide_model.pkl')
preprocessor = pipeline.named_steps['preprocess']
rf_model     = pipeline.named_steps['model']

# Transform test set through the preprocessor
X_test_transformed  = preprocessor.transform(X_test)
X_train_transformed = preprocessor.transform(X_train)

# ─────────────────────────────────────────────
# 3. COMPUTE SHAP VALUES
# ─────────────────────────────────────────────
print("Computing SHAP values (TreeExplainer) …")
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_transformed)

# shap_values shape can be (n, features, classes) or list of (n, features)
# Always extract class-1 slice (Funding = Yes)
if isinstance(shap_values, list):
    sv = shap_values[1]                     # list format
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]              # 3-D array format
else:
    sv = shap_values                        # already 2-D

# Expected value for class 1
ev1 = (explainer.expected_value[1]
       if hasattr(explainer.expected_value, '__len__')
       else explainer.expected_value)

feature_labels = [FEAT_LABELS[f] for f in ALL_FEATS]

# ─────────────────────────────────────────────
# 4. EXPORT RAW SHAP VALUES
# ─────────────────────────────────────────────
shap_df = pd.DataFrame(sv, columns=feature_labels, index=X_test.index)
shap_df.insert(0, 'actual',    y_test.values)
shap_df.insert(1, 'predicted', pipeline.predict(X_test))
shap_df.insert(2, 'prob_funding', pipeline.predict_proba(X_test)[:, 1].round(4))
shap_df.to_csv('shap_values.csv', index=False)
print("✅  shap_values.csv exported")

# ─────────────────────────────────────────────
# 5. PLOT 1 — SUMMARY BAR  (mean |SHAP|)
# ─────────────────────────────────────────────
print("Plotting summary bar …")
mean_abs = np.abs(sv).mean(axis=0)
order     = np.argsort(mean_abs)
sorted_labels = [feature_labels[i] for i in order]
sorted_vals   = mean_abs[order]

bar_colors = [ACCENT1 if v == sorted_vals.max() else ACCENT3 if v >= np.percentile(sorted_vals, 66)
              else '#9575CD' for v in sorted_vals]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)
bars = ax.barh(sorted_labels, sorted_vals, color=bar_colors, edgecolor='none', height=0.6)
for bar, val in zip(bars, sorted_vals):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left', color=TEXT_COL, fontsize=9)
ax.set_xlabel('Mean |SHAP value|  (impact on model output)', fontsize=11)
ax.set_title('SHAP Feature Importance\n(Mean Absolute SHAP — Funding = Yes)',
             fontsize=13, fontweight='bold', pad=14)
for sp in ax.spines.values(): sp.set_color(GRID_COL); sp.set_linewidth(0.5)
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("✅  shap_summary_bar.png saved")

# ─────────────────────────────────────────────
# 6. PLOT 2 — BEESWARM  (SHAP summary plot)
# ─────────────────────────────────────────────
print("Plotting beeswarm …")
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(CARD_BG)

shap.summary_plot(
    sv,
    X_test_transformed,
    feature_names=feature_labels,
    plot_type='dot',
    show=False,
    color_bar=True,
    max_display=8,
    plot_size=None,
)
plt.gcf().set_facecolor(DARK_BG)
plt.gca().set_facecolor(CARD_BG)
plt.title('SHAP Beeswarm — Funding Prediction\n(red = high feature value  |  blue = low feature value)',
          color=TEXT_COL, fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('shap_summary_beeswarm.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("✅  shap_summary_beeswarm.png saved")

# ─────────────────────────────────────────────
# 7. PLOT 3 — DEPENDENCE PLOTS (top 4 features)
# ─────────────────────────────────────────────
print("Plotting dependence grid …")
top4_idx = np.argsort(mean_abs)[::-1][:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('SHAP Dependence Plots — Top 4 Features', color=TEXT_COL,
             fontsize=14, fontweight='bold', y=1.01)

for ax, feat_idx in zip(axes.flat, top4_idx):
    ax.set_facecolor(CARD_BG)
    shap.dependence_plot(
        feat_idx,
        sv,
        X_test_transformed,
        feature_names=feature_labels,
        ax=ax,
        show=False,
        dot_size=25,
        alpha=0.7,
    )
    ax.set_title(f'{feature_labels[feat_idx]}', color=TEXT_COL, fontsize=12, fontweight='bold')
    ax.title.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    for sp in ax.spines.values(): sp.set_color(GRID_COL)
    ax.tick_params(colors=TEXT_COL)

plt.tight_layout()
plt.savefig('shap_dependence_grid.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("✅  shap_dependence_grid.png saved")

# ─────────────────────────────────────────────
# 8. PLOT 4 — WATERFALL for one TP & one TN
# shap.waterfall_plot always creates its own figure internally,
# so we render each into a buffer then composite side-by-side.
# ─────────────────────────────────────────────
print("Plotting waterfall examples …")
import io
from PIL import Image as PILImage

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_arr   = y_test.values

tp_mask = (y_arr == 1) & (y_pred == 1)
tn_mask = (y_arr == 0) & (y_pred == 0)
tp_idx  = np.where(tp_mask)[0][np.argmax(y_proba[tp_mask])]
tn_idx  = np.where(tn_mask)[0][np.argmin(y_proba[tn_mask])]

cases = [
    (tp_idx, f'True Positive — FUNDED\nPredicted probability: {y_proba[tp_idx]:.2%}'),
    (tn_idx, f'True Negative — NOT FUNDED\nPredicted probability: {y_proba[tn_idx]:.2%}'),
]

buffers = []
for idx, title in cases:
    ev = shap.Explanation(
        values        = sv[idx],
        base_values   = ev1,
        data          = X_test_transformed[idx],
        feature_names = feature_labels,
    )
    # Let SHAP create its own figure
    shap.waterfall_plot(ev, show=False, max_display=8)
    wf_fig = plt.gcf()
    wf_fig.patch.set_facecolor(DARK_BG)
    wf_fig.set_size_inches(8, 6)
    # Style the axes SHAP created
    for wf_ax in wf_fig.get_axes():
        wf_ax.set_facecolor(CARD_BG)
        wf_ax.tick_params(colors=TEXT_COL)
        for sp in wf_ax.spines.values():
            sp.set_color(GRID_COL)
    wf_fig.suptitle(title, color=TEXT_COL, fontsize=11, fontweight='bold', y=1.01)
    buf = io.BytesIO()
    wf_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    buf.seek(0)
    buffers.append(PILImage.open(buf).copy())
    plt.close(wf_fig)

# Composite side-by-side
w1, h1 = buffers[0].size
w2, h2 = buffers[1].size
combined_h = max(h1, h2)
combined_w = w1 + w2 + 20          # 20px gap

# Dark background canvas
canvas_arr = np.full((combined_h, combined_w, 3), 15, dtype=np.uint8)   # #0f0f0f ≈ DARK_BG
canvas = PILImage.fromarray(canvas_arr)
canvas.paste(buffers[0], (0,        (combined_h - h1) // 2))
canvas.paste(buffers[1], (w1 + 20,  (combined_h - h2) // 2))

# Add title banner via matplotlib
title_fig, title_ax = plt.subplots(figsize=(combined_w/150, 0.5))
title_fig.patch.set_facecolor(DARK_BG)
title_ax.axis('off')
title_ax.text(0.5, 0.5, 'SHAP Waterfall — Individual Predictions',
              ha='center', va='center', color=TEXT_COL,
              fontsize=14, fontweight='bold', transform=title_ax.transAxes)
title_buf = io.BytesIO()
title_fig.savefig(title_buf, format='png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
title_buf.seek(0)
title_img = PILImage.open(title_buf)
plt.close(title_fig)

tw, th = title_img.size
final_w = max(combined_w, tw)
final_h = combined_h + th
final_arr = np.full((final_h, final_w, 3), 15, dtype=np.uint8)
final_canvas = PILImage.fromarray(final_arr)
final_canvas.paste(title_img,  ((final_w - tw) // 2, 0))
final_canvas.paste(canvas,     ((final_w - combined_w) // 2, th))

final_canvas.save('shap_waterfall_examples.png', dpi=(150, 150))
print("✅  shap_waterfall_examples.png saved")

# ─────────────────────────────────────────────
# 9. CONSOLE SUMMARY
# ─────────────────────────────────────────────
print("\n═══════════════════════════════════════════")
print("  SHAP GLOBAL FEATURE RANKING")
print("═══════════════════════════════════════════")
rank_df = pd.DataFrame({
    'Feature':        feature_labels,
    'Mean |SHAP|':    mean_abs.round(5),
    'Positive pushes':  (sv > 0).sum(axis=0),
    'Negative pushes':  (sv < 0).sum(axis=0),
}).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
rank_df.index += 1
print(rank_df.to_string())

print("\n✅  All SHAP outputs saved:")
print("    shap_summary_bar.png")
print("    shap_summary_beeswarm.png")
print("    shap_dependence_grid.png")
print("    shap_waterfall_examples.png")
print("    shap_values.csv")