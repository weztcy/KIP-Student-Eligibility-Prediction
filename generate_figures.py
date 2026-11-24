import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

# Pastikan folder figures ada
os.makedirs('figures', exist_ok=True)

# Set visual style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 15

# Load dataset
df = pd.read_csv('dataset.csv')

# Warna konsisten: Tidak (Biru Navy/Abu-abu), Ya (Oranye/Kuning Emas)
palette_kip = {'Tidak': '#2563eb', 'Ya': '#f97316'}

# ==============================================================================
# 1. Distribusi Target Penerima KIP
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

counts = df['penerima_kip'].value_counts()
pcts = df['penerima_kip'].value_counts(normalize=True) * 100
bars = axes[0].bar(counts.index, counts.values, color=[palette_kip[k] for k in counts.index], width=0.45, edgecolor='black', linewidth=1.2)
for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2, yval + 8, f'{int(yval)} siswa\n({yval/len(df)*100:.1f}%)', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_title('Jumlah Siswa Berdasarkan Status KIP')
axes[0].set_xlabel('Status Penerima KIP')
axes[0].set_ylabel('Jumlah Siswa')
axes[0].set_ylim(0, max(counts.values) * 1.2)

wedges, texts, autotexts = axes[1].pie(
    counts.values, 
    labels=[f'{k} ({v} Siswa)' for k, v in counts.items()], 
    autopct='%1.1f%%', 
    startangle=140,
    colors=[palette_kip[k] for k in counts.index],
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
    at.set_color('white')
axes[1].set_title('Proporsi Persentase Penerima KIP')

plt.suptitle('Analisis Distribusi Target Penerima KIP (Total: 500 Siswa)', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/01_distribusi_target_kip.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 2. Distribusi Penghasilan Orang Tua
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.kdeplot(data=df, x='penghasilan', hue='penerima_kip', fill=True, common_norm=False, 
            palette=palette_kip, alpha=0.4, linewidth=2, ax=axes[0])
axes[0].set_title('Estimasi Densitas Penghasilan Orang Tua')
axes[0].set_xlabel('Penghasilan Orang Tua (Rp)')
axes[0].set_ylabel('Densitas')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"Rp {int(x):,}"))
axes[0].tick_params(axis='x', rotation=20)

sns.boxplot(data=df, x='penerima_kip', y='penghasilan', hue='penerima_kip', legend=False, palette=palette_kip, ax=axes[1], width=0.4, fliersize=5)
means = df.groupby('penerima_kip')['penghasilan'].mean()
for i, status in enumerate(['Tidak', 'Ya']):
    mean_val = means[status]
    axes[1].text(i, mean_val, f'  Rata-rata:\n  Rp {mean_val:,.0f}', color='black', fontweight='bold', fontsize=10, va='center')

axes[1].set_title('Distribusi Rentang Penghasilan Orang Tua')
axes[1].set_xlabel('Status Penerima KIP')
axes[1].set_ylabel('Penghasilan Orang Tua (Rp)')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"Rp {int(y):,}"))

plt.suptitle('Hubungan Antara Tingkat Penghasilan Orang Tua dan Status KIP', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/02_distribusi_penghasilan_ortu.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 3. Pengaruh Kepemilikan KPS dan Rekomendasi BSM
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

kps_cross = pd.crosstab(df['peserta_kps'], df['penerima_kip'], normalize='index') * 100
kps_counts = pd.crosstab(df['peserta_kps'], df['penerima_kip'])
kps_cross.plot(kind='bar', stacked=True, color=[palette_kip['Tidak'], palette_kip['Ya']], ax=axes[0], edgecolor='black', width=0.5)
axes[0].set_title('Proporsi Penerima KIP Berdasarkan Kepemilikan KPS')
axes[0].set_xlabel('Peserta KPS')
axes[0].set_ylabel('Persentase (%)')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Penerima KIP', loc='upper right')
for idx, (idx_label, row) in enumerate(kps_cross.iterrows()):
    cum = 0
    for col in ['Tidak', 'Ya']:
        val = row[col]
        cnt = kps_counts.loc[idx_label, col]
        if val > 5:
            axes[0].text(idx, cum + val/2, f'{val:.1f}%\n(n={cnt})', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        cum += val

bsm_cross = pd.crosstab(df['rekomendasi_bsm'], df['penerima_kip'], normalize='index') * 100
bsm_counts = pd.crosstab(df['rekomendasi_bsm'], df['penerima_kip'])
bsm_cross.plot(kind='bar', stacked=True, color=[palette_kip['Tidak'], palette_kip['Ya']], ax=axes[1], edgecolor='black', width=0.5)
axes[1].set_title('Proporsi Penerima KIP Berdasarkan Rekomendasi BSM')
axes[1].set_xlabel('Mendapat Rekomendasi BSM')
axes[1].set_ylabel('Persentase (%)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='Penerima KIP', loc='upper right')
for idx, (idx_label, row) in enumerate(bsm_cross.iterrows()):
    cum = 0
    for col in ['Tidak', 'Ya']:
        val = row[col]
        cnt = bsm_counts.loc[idx_label, col]
        if val > 5:
            axes[1].text(idx, cum + val/2, f'{val:.1f}%\n(n={cnt})', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        cum += val

plt.suptitle('Pengaruh Program Bantuan Sebelumnya (KPS & BSM) Terhadap Kelayakan KIP', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/03_pengaruh_kps_dan_bsm.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 4. Pengaruh Moda Transportasi Siswa
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.countplot(data=df, x='alat_transportasi', hue='penerima_kip', palette=palette_kip, 
              order=['Jalan kaki', 'Angkutan umum', 'Kendaraan pribadi'], edgecolor='black', ax=axes[0])
axes[0].set_title('Jumlah Siswa per Alat Transportasi & Status KIP')
axes[0].set_xlabel('Alat Transportasi ke Sekolah')
axes[0].set_ylabel('Jumlah Siswa')
axes[0].legend(title='Penerima KIP')
for p in axes[0].patches:
    height = p.get_height()
    if height > 0:
        axes[0].text(p.get_x() + p.get_width()/2., height + 2, f'{int(height)}', ha="center", fontsize=10, fontweight='bold')

trans_rate = df.groupby('alat_transportasi')['penerima_kip'].apply(lambda s: (s == 'Ya').mean() * 100).reindex(['Jalan kaki', 'Angkutan umum', 'Kendaraan pribadi'])
bars = axes[1].bar(trans_rate.index, trans_rate.values, color=['#e11d48', '#f59e0b', '#10b981'], edgecolor='black', width=0.5)
for bar in bars:
    y = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, y + 0.8, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=11)
axes[1].set_title('Tingkat Penerimaan KIP (%) Berdasarkan Transportasi')
axes[1].set_xlabel('Alat Transportasi ke Sekolah')
axes[1].set_ylabel('Persentase Diterima KIP (%)')
axes[1].set_ylim(0, max(trans_rate.values) * 1.25)

plt.suptitle('Analisis Moda Transportasi Siswa dan Relevansinya dengan KIP', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/04_pengaruh_alat_transportasi.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 5. Pengaruh Jumlah Saudara Kandung & Jarak Rumah
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

saudara_cross = pd.crosstab(df['jumlah_saudara_kandung'], df['penerima_kip'], normalize='index') * 100
saudara_cross.plot(kind='bar', stacked=True, color=[palette_kip['Tidak'], palette_kip['Ya']], ax=axes[0], edgecolor='black', width=0.5)
axes[0].set_title('Status KIP Berdasarkan Jumlah Saudara Kandung')
axes[0].set_xlabel('Jumlah Saudara Kandung')
axes[0].set_ylabel('Persentase (%)')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='Penerima KIP', loc='upper right')
for idx, (idx_label, row) in enumerate(saudara_cross.iterrows()):
    cum = 0
    for col in ['Tidak', 'Ya']:
        val = row[col]
        if val > 5:
            axes[0].text(idx, cum + val/2, f'{val:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        cum += val

jarak_cross = pd.crosstab(df['jarak_rumah_ke_sekolah'], df['penerima_kip'], normalize='index') * 100
jarak_cross.plot(kind='bar', stacked=True, color=[palette_kip['Tidak'], palette_kip['Ya']], ax=axes[1], edgecolor='black', width=0.5)
axes[1].set_title('Status KIP Berdasarkan Jarak Rumah ke Sekolah')
axes[1].set_xlabel('Jarak Rumah ke Sekolah (km / kategori skala)')
axes[1].set_ylabel('Persentase (%)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='Penerima KIP', loc='upper right')
for idx, (idx_label, row) in enumerate(jarak_cross.iterrows()):
    cum = 0
    for col in ['Tidak', 'Ya']:
        val = row[col]
        if val > 5:
            axes[1].text(idx, cum + val/2, f'{val:.1f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        cum += val

plt.suptitle('Analisis Faktor Beban Tanggungan Keluarga & Aksesibilitas Jarak', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/05_pengaruh_saudara_dan_jarak.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 6. Korelasi Antar Fitur (Heatmap)
# ==============================================================================
df_encoded = df.copy()
encode_cols = ['peserta_kps', 'alat_transportasi', 'rekomendasi_bsm']
for col in encode_cols:
    dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype=int)
    df_encoded = pd.concat([df_encoded, dummies], axis=1)
    del df_encoded[col]

df_encoded['penerima_kip'] = df_encoded['penerima_kip'].map({'Tidak': 0, 'Ya': 1})
corr = df_encoded.corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt='.2f', 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriks Korelasi Pearson Fitur dan Target Penerima KIP', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('figures/06_korelasi_fitur_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 7. Evaluasi Model Gaussian Naive Bayes: Confusion Matrix & Analisis Probabilitas
# ==============================================================================
X = df_encoded.drop(columns=['penerima_kip'])
y = df_encoded['penerima_kip']

# Load model asli
model = pickle.load(open('modelNBC_kip.pkl', 'rb'))
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

cm_default = confusion_matrix(y, y_pred)

# Cari threshold optimal berdasarkan Youden's Index pada ROC
fpr, tpr, thresholds = roc_curve(y, y_proba)
optimal_idx = np.argmax(tpr - fpr)
opt_threshold = thresholds[optimal_idx]
y_pred_opt = (y_proba >= opt_threshold).astype(int)
cm_opt = confusion_matrix(y, y_pred_opt)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Confusion Matrix Default Threshold (0.50)
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Prediksi Tidak', 'Prediksi Ya'],
            yticklabels=['Aktual Tidak', 'Aktual Ya'], annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_title('Confusion Matrix Model Asli (Threshold Standar: 0.50)\nCatatan: Terjadi Imbalance & Prior Effect')
axes[0].set_ylabel('Aktual')
axes[0].set_xlabel('Prediksi Model')

# Confusion Matrix Optimal Threshold
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
            xticklabels=['Prediksi Tidak', 'Prediksi Ya'],
            yticklabels=['Aktual Tidak', 'Aktual Ya'], annot_kws={'size': 14, 'weight': 'bold'})
axes[1].set_title(f'Confusion Matrix dengan Threshold Optimal ({opt_threshold:.3f})\nSensitivitas: {tpr[optimal_idx]*100:.1f}%, Spesifisitas: {(1-fpr[optimal_idx])*100:.1f}%')
axes[1].set_ylabel('Aktual')
axes[1].set_xlabel('Prediksi Model')

plt.suptitle('Evaluasi Performa Model Gaussian Naive Bayes (modelNBC_kip.pkl)', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/07_evaluasi_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 8. Kurva ROC dan Precision-Recall
# ==============================================================================
roc_auc = auc(fpr, tpr)
prec, rec, _ = precision_recall_curve(y, y_proba)
pr_auc = average_precision_score(y, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
axes[0].plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
axes[0].scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=80, zorder=5, label=f'Titik Optimal (Thresh={opt_threshold:.3f})')
axes[0].fill_between(fpr, tpr, color='#93c5fd', alpha=0.3)
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate (1 - Spesifisitas)')
axes[0].set_ylabel('True Positive Rate (Sensitivitas / Recall)')
axes[0].set_title('Receiver Operating Characteristic (ROC)')
axes[0].legend(loc="lower right", fontsize=10)

# PR Curve
axes[1].plot(rec, prec, color='#dc2626', lw=2.5, label=f'PR Curve (AP = {pr_auc:.3f})')
axes[1].axhline(y=y.mean(), color='gray', linestyle='--', label=f'Baseline Tanpa Model ({y.mean():.3f})')
axes[1].fill_between(rec, prec, color='#fca5a5', alpha=0.3)
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('Recall (Kemampuan Menemukan Siswa Layak)')
axes[1].set_ylabel('Precision (Ketepatan Prediksi Siswa Layak)')
axes[1].set_title('Precision-Recall Curve (PR)')
axes[1].legend(loc="upper right", fontsize=10)

plt.suptitle('Analisis Diskriminasi Probabilitas Model Gaussian Naive Bayes', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/08_evaluasi_model_roc_pr_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 9. Analisis Multivariat: Penghasilan, Jarak, Transportasi & KIP
# ==============================================================================
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x='jarak_rumah_ke_sekolah',
    y='penghasilan',
    hue='penerima_kip',
    style='alat_transportasi',
    palette=palette_kip,
    s=120,
    alpha=0.8,
    edgecolor='black'
)
plt.title('Hubungan Multivariat: Jarak Sekolah vs Penghasilan vs Alat Transportasi & Status KIP', pad=15)
plt.xlabel('Jarak Rumah ke Sekolah (km / skala)', labelpad=10)
plt.ylabel('Penghasilan Orang Tua (Rp)', labelpad=10)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, loc: f"Rp {int(y):,}"))
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Atribut')
plt.tight_layout()
plt.savefig('figures/09_multivariate_penghasilan_jarak_kip.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# 10. Profil Karakteristik Rata-rata Penerima vs Bukan Penerima KIP
# ==============================================================================
profile_summary = pd.DataFrame({
    'Metrik / Fitur': [
        'Rata-rata Penghasilan (Juta Rp)',
        'Proporsi Peserta KPS (%)',
        'Proporsi Rekomendasi BSM (%)',
        'Proporsi Jalan Kaki (%)',
        'Rata-rata Saudara Kandung',
        'Rata-rata Jarak ke Sekolah (km)'
    ],
    'Penerima KIP (Ya)': [
        df[df['penerima_kip']=='Ya']['penghasilan'].mean() / 1e6,
        (df[df['penerima_kip']=='Ya']['peserta_kps'] == 'Ya').mean() * 100,
        (df[df['penerima_kip']=='Ya']['rekomendasi_bsm'] == 'Ya').mean() * 100,
        (df[df['penerima_kip']=='Ya']['alat_transportasi'] == 'Jalan kaki').mean() * 100,
        df[df['penerima_kip']=='Ya']['jumlah_saudara_kandung'].mean(),
        df[df['penerima_kip']=='Ya']['jarak_rumah_ke_sekolah'].mean()
    ],
    'Bukan Penerima (Tidak)': [
        df[df['penerima_kip']=='Tidak']['penghasilan'].mean() / 1e6,
        (df[df['penerima_kip']=='Tidak']['peserta_kps'] == 'Ya').mean() * 100,
        (df[df['penerima_kip']=='Tidak']['rekomendasi_bsm'] == 'Ya').mean() * 100,
        (df[df['penerima_kip']=='Tidak']['alat_transportasi'] == 'Jalan kaki').mean() * 100,
        df[df['penerima_kip']=='Tidak']['jumlah_saudara_kandung'].mean(),
        df[df['penerima_kip']=='Tidak']['jarak_rumah_ke_sekolah'].mean()
    ]
})

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(profile_summary))
width = 0.35

rects1 = ax.bar(x - width/2, profile_summary['Penerima KIP (Ya)'], width, label='Penerima KIP (Ya)', color='#f97316', edgecolor='black')
rects2 = ax.bar(x + width/2, profile_summary['Bukan Penerima (Tidak)'], width, label='Bukan Penerima (Tidak)', color='#2563eb', edgecolor='black')

ax.set_title('Perbandingan Komprehensif Profil Siswa Penerima vs Bukan Penerima KIP')
ax.set_xticks(x)
ax.set_xticklabels(profile_summary['Metrik / Fitur'], rotation=25, ha='right', fontweight='semibold')
ax.set_ylabel('Nilai / Persentase')
ax.legend(loc='upper right')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}', (rect.get_x() + rect.get_width() / 2., h),
                ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}', (rect.get_x() + rect.get_width() / 2., h),
                ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.savefig('figures/10_perbandingan_profil_penerima_kip.png', dpi=300, bbox_inches='tight')
plt.close()

print("Semua visualisasi berhasil digenerate ke folder 'figures' tanpa error!")
