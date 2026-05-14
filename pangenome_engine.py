import numpy as np; np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import shutil, os

# ── Parameters ──────────────────────────────────────────────────────────────
N_GENOMES = 100
N_GENES_TOTAL = 8000   # total pan-genome size

# ── Gene presence/absence matrix ─────────────────────────────────────────────
# Each gene has a frequency across genomes
gene_freq = np.random.beta(0.5, 0.5, N_GENES_TOTAL)
# Core: >95%, Accessory: 15-95%, Unique: <15%
gene_freq = np.clip(gene_freq, 0.01, 0.99)

presence = np.zeros((N_GENOMES, N_GENES_TOTAL), dtype=np.int8)
for g in range(N_GENES_TOTAL):
    presence[:, g] = np.random.binomial(1, gene_freq[g], N_GENOMES)

# Classify genes
freq_across = presence.mean(axis=0)
core_mask   = freq_across > 0.95
access_mask = (freq_across >= 0.15) & (freq_across <= 0.95)
unique_mask = freq_across < 0.15

n_core   = core_mask.sum()
n_access = access_mask.sum()
n_unique = unique_mask.sum()

# ── Pan-genome accumulation curve ────────────────────────────────────────────
pan_sizes  = []
core_sizes = []
order = np.random.permutation(N_GENOMES)
seen_genes = np.zeros(N_GENES_TOTAL, dtype=bool)
core_genes = np.ones(N_GENES_TOTAL, dtype=bool)

for i, g in enumerate(order):
    seen_genes |= presence[g].astype(bool)
    core_genes &= presence[g].astype(bool)
    pan_sizes.append(seen_genes.sum())
    core_sizes.append(core_genes.sum())

pan_sizes  = np.array(pan_sizes)
core_sizes = np.array(core_sizes)
genome_nums = np.arange(1, N_GENOMES+1)

# ── Heaps' law fitting ────────────────────────────────────────────────────────
def heaps_law(n, kappa, gamma):
    return kappa * n**gamma

try:
    popt, _ = curve_fit(heaps_law, genome_nums, pan_sizes, p0=[1000, 0.3], maxfev=5000)
    kappa, gamma_heap = popt
except:
    kappa, gamma_heap = 1000, 0.3

pan_fit = heaps_law(genome_nums, kappa, gamma_heap)

# ── Variation graph stats ─────────────────────────────────────────────────────
n_snps_vg  = np.random.poisson(50000, N_GENOMES)
n_indels   = np.random.poisson(5000, N_GENOMES)
n_svs      = np.random.poisson(500, N_GENOMES)
genome_sizes = np.random.normal(4.5e6, 0.3e6, N_GENOMES)   # ~4.5 Mb bacterial genome

# ── Functional enrichment of accessory genes ─────────────────────────────────
func_cats = ['Metabolism', 'Virulence', 'Resistance', 'Mobility', 'Hypothetical',
             'Transport', 'Regulation', 'Biosynthesis']
# Accessory genes enriched for virulence/resistance/mobility
access_enrich = np.array([0.15, 0.20, 0.18, 0.15, 0.12, 0.08, 0.07, 0.05])
core_enrich   = np.array([0.35, 0.05, 0.03, 0.02, 0.10, 0.15, 0.15, 0.15])
access_enrich /= access_enrich.sum()
core_enrich   /= core_enrich.sum()

# ── Accessory gene clustering ─────────────────────────────────────────────────
# PCA of accessory gene presence/absence
access_mat = presence[:, access_mask].astype(float)
if access_mat.shape[1] > 0:
    A = access_mat - access_mat.mean(axis=0)
    cov_a = A @ A.T / access_mat.shape[1]
    eigvals_a, eigvecs_a = np.linalg.eigh(cov_a)
    idx_a = np.argsort(eigvals_a)[::-1]
    PC_access = eigvecs_a[:, idx_a[:2]]
    var_access = eigvals_a[idx_a[:2]] / eigvals_a.sum() * 100
else:
    PC_access = np.random.randn(N_GENOMES, 2)
    var_access = [10, 5]

# ── Dashboard ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Pan-Genome Engine — Dashboard', color='white', fontsize=16, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel(xlabel, color='#8b949e')
    ax.set_ylabel(ylabel, color='#8b949e')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1 — Pan-genome accumulation curve
ax = axes[0,0]
ax.plot(genome_nums, pan_sizes, color='#58a6ff', lw=2.5, label='Pan-genome')
ax.plot(genome_nums, pan_fit, color='#ffa657', lw=2, ls='--',
        label=f"Heaps' law (γ={gamma_heap:.2f})")
ax.plot(genome_nums, core_sizes, color='#3fb950', lw=2.5, label='Core genome')
style_ax(ax, "Pan-Genome Accumulation Curve", 'Number of Genomes', 'Gene Count')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 2 — Core/Accessory/Unique pie
ax = axes[0,1]
sizes = [n_core, n_access, n_unique]
labels = [f'Core\n({n_core:,})', f'Accessory\n({n_access:,})', f'Unique\n({n_unique:,})']
colors_pie = ['#3fb950', '#58a6ff', '#f78166']
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                   autopct='%1.1f%%', startangle=90,
                                   textprops={'color': 'white', 'fontsize': 9})
for at in autotexts:
    at.set_color('white')
style_ax(ax, 'Pan-Genome Composition')

# Panel 3 — Heaps' law fit
ax = axes[0,2]
ax.scatter(genome_nums[::5], pan_sizes[::5], c='#58a6ff', s=30, alpha=0.7, label='Observed')
ax.plot(genome_nums, pan_fit, color='#f78166', lw=2.5,
        label=f"Heaps' law\nκ={kappa:.0f}, γ={gamma_heap:.3f}")
style_ax(ax, "Heaps' Law Fit", 'Genomes', 'Pan-genome Size')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 4 — Gene presence/absence heatmap (subset)
ax = axes[1,0]
# Show 50 genomes × 200 accessory genes
sub_pres = presence[:50, access_mask][:, :200] if access_mask.sum() >= 200 else presence[:50, access_mask]
im = ax.imshow(sub_pres, aspect='auto', cmap='Blues', interpolation='nearest')
style_ax(ax, 'Gene Presence/Absence Matrix', 'Accessory Gene', 'Genome')
plt.colorbar(im, ax=ax, label='Present')

# Panel 5 — Functional enrichment
ax = axes[1,1]
x = np.arange(len(func_cats))
w = 0.35
ax.bar(x - w/2, core_enrich*100, w, color='#3fb950', label='Core', alpha=0.85)
ax.bar(x + w/2, access_enrich*100, w, color='#58a6ff', label='Accessory', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(func_cats, rotation=35, ha='right', color='white', fontsize=8)
style_ax(ax, 'Functional Enrichment (Accessory vs Core)', 'Function', '% of Genes')
ax.legend(fontsize=8, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 6 — Variation graph stats
ax = axes[1,2]
vg_data = [n_snps_vg/1000, n_indels/1000, n_svs]
vg_labels = ['SNPs (k)', 'Indels (k)', 'SVs']
vg_colors = ['#58a6ff', '#3fb950', '#f78166']
bp = ax.boxplot(vg_data, labels=vg_labels, patch_artist=True,
                medianprops={'color': 'white', 'lw': 2})
for patch, color in zip(bp['boxes'], vg_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp[element]:
        item.set_color('#8b949e')
style_ax(ax, 'Variation Graph Statistics', 'Variant Type', 'Count')

# Panel 7 — Genome size distribution
ax = axes[2,0]
ax.hist(genome_sizes/1e6, bins=25, color='#ffa657', edgecolor='#0d1117', alpha=0.85)
ax.axvline(genome_sizes.mean()/1e6, color='white', lw=2, ls='--',
           label=f'Mean={genome_sizes.mean()/1e6:.2f} Mb')
style_ax(ax, 'Genome Size Distribution', 'Genome Size (Mb)', 'Count')
ax.legend(fontsize=9, labelcolor='white', facecolor='#21262d', edgecolor='#30363d')

# Panel 8 — Accessory gene clustering (PCA)
ax = axes[2,1]
sc = ax.scatter(PC_access[:,0], PC_access[:,1], c=genome_sizes/1e6,
                cmap='viridis', s=50, alpha=0.8, edgecolors='#30363d')
plt.colorbar(sc, ax=ax, label='Genome Size (Mb)')
style_ax(ax, f'Accessory Genome Clustering (PCA)\nPC1={var_access[0]:.1f}%, PC2={var_access[1]:.1f}%',
         'PC1', 'PC2')

# Panel 9 — Summary
ax = axes[2,2]
ax.axis('off')
style_ax(ax, 'Summary Statistics')
summary = [
    f'Bacterial genomes: {N_GENOMES}',
    f'Total pan-genome: {N_GENES_TOTAL:,} genes',
    f'Core genome: {n_core:,} ({n_core/N_GENES_TOTAL*100:.1f}%)',
    f'Accessory genome: {n_access:,} ({n_access/N_GENES_TOTAL*100:.1f}%)',
    f'Unique genes: {n_unique:,} ({n_unique/N_GENES_TOTAL*100:.1f}%)',
    f"Heaps' law γ: {gamma_heap:.3f}",
    f'Mean genome size: {genome_sizes.mean()/1e6:.2f} Mb',
    f'Mean SNPs/genome: {n_snps_vg.mean():.0f}',
    f'Mean SVs/genome: {n_svs.mean():.0f}',
    f'Open pan-genome: {gamma_heap > 0}',
]
for k, line in enumerate(summary):
    ax.text(0.05, 0.92 - k*0.09, line, transform=ax.transAxes,
            color='#e6edf3', fontsize=10, va='top')

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = '/mnt/shared-workspace/shared/pangenome_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f'Dashboard saved: {out_png}')

shutil.copy('/workspace/subagents/a29c645f/pangenome_engine.py',
            '/mnt/shared-workspace/shared/pangenome_engine.py')

print('\n=== KEY RESULTS: PangenomeEngine ===')
print(f'Total pan-genome: {N_GENES_TOTAL:,} genes')
print(f'Core genome: {n_core:,} ({n_core/N_GENES_TOTAL*100:.1f}%)')
print(f'Accessory genome: {n_access:,} ({n_access/N_GENES_TOTAL*100:.1f}%)')
print(f'Unique genes: {n_unique:,} ({n_unique/N_GENES_TOTAL*100:.1f}%)')
print(f"Heaps' law: κ={kappa:.0f}, γ={gamma_heap:.3f}")
print(f'Mean genome size: {genome_sizes.mean()/1e6:.2f} Mb')
print(f'Open pan-genome (γ>0): {gamma_heap > 0}')
