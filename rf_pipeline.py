import pandas as pd
import numpy as np
import re
import warnings
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_excel('Tide_2_0_final.xlsx')

# ─────────────────────────────────────────────
# 2. CLEAN TARGET — Whether Follow-on Funding Received → binary 0/1
# ─────────────────────────────────────────────
def clean_target(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ('yes', 'y'):
        return 1
    if s in ('no', 'n', '-', '0'):
        return 0
    # paragraphs that mention grant/funding → 1
    keywords = ['funding', 'grant', 'investment', 'investor', 'crore', 'lakh', 'received']
    if any(k in s for k in keywords):
        return 1
    return np.nan

df['target'] = df['Whether Follow-on Funding Received'].apply(clean_target)

# ─────────────────────────────────────────────
# 3. CLEAN FEATURES
# ─────────────────────────────────────────────

# --- Amount sanctioned → numeric
def parse_amount(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(',', '').replace('₹', '').replace('rs', '').replace('inr', '').strip()
    # Handle "X lakhs / lacs" patterns
    m = re.match(r'([\d.]+)\s*(lakh|lac|l)', s)
    if m:
        return float(m.group(1)) * 100000
    m = re.match(r'([\d.]+)\s*(crore|cr)', s)
    if m:
        return float(m.group(1)) * 10000000
    # plain number
    try:
        return float(s)
    except:
        return np.nan

df['amount_sanctioned'] = df['Amount sanctioned under the scheme'].apply(parse_amount)

# --- Team size → numeric
def parse_team(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip().split()[0])
    except:
        return np.nan

df['team_size'] = df['Team size at the time of onboarding'].apply(parse_team)
df['team_size'] = df['team_size'].clip(0, 100)  # remove outliers

# --- TRL → numeric 1–9
def parse_trl(val):
    if pd.isna(val):
        return np.nan
    m = re.search(r'\d+', str(val))
    if m:
        v = float(m.group())
        return v if 1 <= v <= 9 else np.nan
    return np.nan

df['trl'] = df['Technology Readiness Level (TRL, at the time of onboarding)'].apply(parse_trl)

# --- Tier normalise
def clean_tier(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if '1' in s or 'I' == s[-1]:
        return 'I'
    if '2' in s or 'II' in s:
        return 'II'
    if '3' in s or 'III' in s:
        return 'III'
    return np.nan

df['tier'] = df['Tier I / II / III'].apply(clean_tier)

# --- Stage normalise
stage_map = {
    'ideation': 'Ideation',
    'proof of concept': 'POC', 'poc': 'POC',
    'prototype': 'Prototype', 'prototype development': 'Prototype',
    'mvp': 'MVP',
    'validation': 'Validation',
    'pilot': 'Pilot',
    'pre-revenue': 'Pre-Revenue', 'pre revenue': 'Pre-Revenue',
    'revenue': 'Revenue',
}
def clean_stage(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    for k, v in stage_map.items():
        if k in s:
            return v
    return 'Other'

df['stage'] = df['Stage of the Startup at the time of onboarding'].apply(clean_stage)

# --- hw/sw normalise
def clean_hwsw(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if 'both' in s:
        return 'Both'
    if 'hard' in s:
        return 'Hardware'
    if 'soft' in s:
        return 'Software'
    return 'Other'

df['hw_sw'] = df['Is it a hardware startup or software startup'].apply(clean_hwsw)

# --- women founder normalise
def clean_women(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    return 'Yes' if s == 'yes' else 'No'

df['women_founder'] = df['Is there a Women Founder/Co-Founder'].apply(clean_women)

# --- Sector: group low-frequency sectors as 'Other'
sector_counts = df['Sector'].value_counts()
top_sectors = sector_counts[sector_counts >= 20].index
df['sector'] = df['Sector'].apply(lambda x: x if x in top_sectors else 'Other')

# ─────────────────────────────────────────────
# 4. EXPORT CLEANED DATA TO EXCEL
# ─────────────────────────────────────────────
cleaned_export = df.copy()

# Replace original messy columns with cleaned versions
cleaned_export['Amount sanctioned under the scheme']                          = df['amount_sanctioned']
cleaned_export['Team size at the time of onboarding']                         = df['team_size']
cleaned_export['Technology Readiness Level (TRL, at the time of onboarding)'] = df['trl']
cleaned_export['Tier I / II / III']                                           = df['tier']
cleaned_export['Stage of the Startup at the time of onboarding']              = df['stage']
cleaned_export['Is it a hardware startup or software startup']                = df['hw_sw']
cleaned_export['Is there a Women Founder/Co-Founder']                         = df['women_founder']
cleaned_export['Sector']                                                       = df['sector']
cleaned_export['Whether Follow-on Funding Received']                          = df['target'].map({1: 'Yes', 0: 'No'})

# Drop the temp helper columns
cleaned_export.drop(columns=['amount_sanctioned','team_size','trl','tier',
                              'stage','hw_sw','women_founder','sector','target'],
                    inplace=True)

# Add a column flagging rows dropped due to unreadable target
cleaned_export.insert(0, 'Row Used in Model', df['target'].apply(
    lambda x: 'Yes' if not pd.isna(x) else 'No — unreadable target'
))

cleaned_export.to_excel('tide_cleaned_data.xlsx', index=False)
print("✅  Cleaned data exported → tide_cleaned_data.xlsx")

# ─────────────────────────────────────────────
# 5. ASSEMBLE FEATURE MATRIX
# ─────────────────────────────────────────────
NUMERIC_FEATS  = ['amount_sanctioned', 'team_size', 'trl']
CATEG_FEATS    = ['sector', 'tier', 'stage', 'hw_sw', 'women_founder']
ALL_FEATS      = NUMERIC_FEATS + CATEG_FEATS

model_df = df[ALL_FEATS + ['target']].dropna(subset=['target'])
print(f"Total rows after target clean: {len(model_df)}")
print(f"Class balance:\n{model_df['target'].value_counts()}")

X = model_df[ALL_FEATS]
y = model_df['target'].astype(int)

# ─────────────────────────────────────────────
# 6. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, NUMERIC_FEATS),
    ('cat', cat_transformer, CATEG_FEATS),
])

# ─────────────────────────────────────────────
# 6. RANDOM FOREST MODEL
# ─────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', rf)
])

# ─────────────────────────────────────────────
# 7. TRAIN / TEST SPLIT + EVALUATE
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

full_pipeline.fit(X_train, y_train)
y_pred  = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

print("\n===== TEST SET RESULTS =====")
print(classification_report(y_test, y_pred, target_names=['No Funding', 'Funding']))
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(full_pipeline, X, y, cv=cv, scoring='roc_auc')
print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
feature_names = NUMERIC_FEATS + CATEG_FEATS
importances   = full_pipeline.named_steps['model'].feature_importances_
feat_imp_df   = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df   = feat_imp_df.sort_values('Importance', ascending=False)

# ─────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────
joblib.dump(full_pipeline, 'rf_tide_model.pkl')

# Save metadata for future predictions
meta = {
    'numeric_features': NUMERIC_FEATS,
    'categorical_features': CATEG_FEATS,
    'all_features': ALL_FEATS,
    'target_column': 'Whether Follow-on Funding Received',
    'model_type': 'RandomForestClassifier',
    'test_accuracy': round(accuracy_score(y_test, y_pred), 4),
    'test_roc_auc': round(roc_auc_score(y_test, y_proba), 4),
    'cv_roc_auc_mean': round(cv_scores.mean(), 4),
    'cv_roc_auc_std': round(cv_scores.std(), 4),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
}
with open('model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

# ─────────────────────────────────────────────
# 10. VISUALISATIONS
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

DARK_BG   = '#0f1117'
CARD_BG   = '#1e2130'
ACCENT1   = '#4C9BE8'
ACCENT2   = '#F06292'
ACCENT3   = '#81C784'
TEXT_COL  = '#E0E0E0'
GRID_COL  = '#2a2d3a'

# ── Plot 1: Feature Importance ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD_BG)
colors_imp = [ACCENT1 if i == 0 else ACCENT3 if i < 3 else '#9575CD' for i in range(len(feat_imp_df))]
bars = ax1.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1],
                color=colors_imp[::-1], edgecolor='none', height=0.6)
for bar, val in zip(bars, feat_imp_df['Importance'][::-1]):
    ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', ha='left', color=TEXT_COL, fontsize=9)
ax1.set_xlabel('Importance Score', color=TEXT_COL, fontsize=10)
ax1.set_title('Feature Importance', color=TEXT_COL, fontsize=13, fontweight='bold', pad=12)
ax1.tick_params(colors=TEXT_COL)
ax1.spines[:].set_color(GRID_COL)
for spine in ax1.spines.values():
    spine.set_linewidth(0.5)

# ── Plot 2: Confusion Matrix ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD_BG)
cm = confusion_matrix(y_test, y_pred)
im = ax2.imshow(cm, cmap='Blues', aspect='auto')
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                 color='white' if cm[i,j] > cm.max()/2 else DARK_BG,
                 fontsize=20, fontweight='bold')
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Pred: No', 'Pred: Yes'], color=TEXT_COL, fontsize=10)
ax2.set_yticklabels(['Actual: No', 'Actual: Yes'], color=TEXT_COL, fontsize=10)
ax2.set_title('Confusion Matrix', color=TEXT_COL, fontsize=13, fontweight='bold', pad=12)
ax2.tick_params(colors=TEXT_COL)
for spine in ax2.spines.values():
    spine.set_edgecolor(GRID_COL)

# ── Plot 3: CV Score Distribution ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD_BG)
folds = [f'Fold {i+1}' for i in range(5)]
bar_colors = [ACCENT1 if s == max(cv_scores) else ACCENT3 for s in cv_scores]
b3 = ax3.bar(folds, cv_scores, color=bar_colors, edgecolor='none', width=0.55)
for bar, v in zip(b3, cv_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{v:.3f}', ha='center', va='bottom', color=TEXT_COL, fontsize=10, fontweight='bold')
ax3.axhline(cv_scores.mean(), color=ACCENT2, linestyle='--', linewidth=1.5, label=f'Mean: {cv_scores.mean():.3f}')
ax3.set_ylim(0, 1.05)
ax3.set_ylabel('ROC-AUC', color=TEXT_COL, fontsize=10)
ax3.set_title('5-Fold Cross-Validation ROC-AUC', color=TEXT_COL, fontsize=13, fontweight='bold', pad=12)
ax3.tick_params(colors=TEXT_COL)
ax3.legend(facecolor=CARD_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=10)
for spine in ax3.spines.values():
    spine.set_color(GRID_COL); spine.set_linewidth(0.5)

# ── Plot 4: Metrics Summary Card ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD_BG)
ax4.axis('off')

metrics = [
    ('Accuracy',         f"{accuracy_score(y_test, y_pred)*100:.1f}%",  ACCENT1),
    ('ROC-AUC (Test)',   f"{roc_auc_score(y_test, y_proba):.4f}",        ACCENT3),
    ('CV ROC-AUC',       f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}", ACCENT2),
    ('Train Samples',    str(len(X_train)),                               '#9575CD'),
    ('Test Samples',     str(len(X_test)),                                '#9575CD'),
    ('n_estimators',     '300',                                           '#F9A825'),
    ('Max Depth',        '12',                                            '#F9A825'),
]
ax4.text(0.5, 0.97, 'Model Report', transform=ax4.transAxes,
         ha='center', va='top', color=TEXT_COL, fontsize=14, fontweight='bold')
for idx, (label, value, color) in enumerate(metrics):
    y_pos = 0.82 - idx * 0.115
    ax4.text(0.12, y_pos, label + ':', transform=ax4.transAxes,
             ha='left', va='center', color='#9E9E9E', fontsize=11)
    ax4.text(0.88, y_pos, value, transform=ax4.transAxes,
             ha='right', va='center', color=color, fontsize=11, fontweight='bold')
    if idx < len(metrics) - 1:
        line_y = y_pos - 0.055
        ax4.plot([0.05, 0.95], [line_y, line_y], color=GRID_COL, linewidth=0.5,
                 transform=ax4.transAxes, clip_on=False)

fig.suptitle('Random Forest — TIDE 2.0 Follow-on Funding Prediction',
             color=TEXT_COL, fontsize=16, fontweight='bold', y=0.98)

plt.savefig('rf_model_report.png',
            dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()

print("\n✅  Model saved  → rf_tide_model.pkl")
print("✅  Report saved → rf_model_report.png")
print("✅  Metadata     → model_metadata.json")
print("\nFeature Importances:")
print(feat_imp_df.to_string(index=False))
