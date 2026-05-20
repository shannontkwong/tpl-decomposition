import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = 'outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

result   = pd.read_csv(os.path.join(OUT_DIR, 'bbs_prospective_extinction.csv'))
result_c1 = pd.read_csv(os.path.join(OUT_DIR, 'bbs_c1_subgroup.csv'))

slope, intercept, *_ = stats.linregress(
    np.log(result['mean_N']), np.log(result['sigma2_L']))
result['resid'] = (
    np.log(result['sigma2_L']) -
    (slope * np.log(result['mean_N']) + intercept)
)
low_all  = result[result['resid'] <  result['resid'].median()]
high_all = result[result['resid'] >= result['resid'].median()]

q25 = result_c1['c1'].quantile(0.25)
q75 = result_c1['c1'].quantile(0.75)

def get_subgroup_rates(sub):
    sl, si, *_ = stats.linregress(
        np.log(sub['mean_N']), np.log(sub['sigma2_L']))
    sub = sub.copy()
    sub['resid_c1'] = (
        np.log(sub['sigma2_L']) - (sl * np.log(sub['mean_N']) + si))
    l = sub[sub['resid_c1'] <  sub['resid_c1'].median()]
    h = sub[sub['resid_c1'] >= sub['resid_c1'].median()]
    _, p = stats.mannwhitneyu(
        l['went_extinct'].astype(int),
        h['went_extinct'].astype(int),
        alternative='two-sided'
    )
    rel = (l['went_extinct'].mean() - h['went_extinct'].mean()) / \
          l['went_extinct'].mean()
    return rel, p

rel_low,  p_low  = get_subgroup_rates(result_c1[result_c1['c1'] <= q25])
rel_high, p_high = get_subgroup_rates(result_c1[result_c1['c1'] >= q75])

plt.rcParams.update({
    'font.family'       : 'sans-serif',
    'font.sans-serif'   : ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size'         : 9,
    'axes.linewidth'    : 0.8,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'xtick.major.width' : 0.8,
    'ytick.major.width' : 0.8,
    'xtick.major.size'  : 3,
    'ytick.major.size'  : 3,
    'pdf.fonttype'      : 42,
    'ps.fonttype'       : 42,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.45, bottom=0.22)

ax = axes[0]

rates  = [low_all['went_extinct'].mean() * 100,
          high_all['went_extinct'].mean() * 100]
colors = ['#C0392B', '#2E86AB']
x      = [0, 1]

bars = ax.bar(x, rates, color=colors, width=0.45,
              edgecolor='none', zorder=3)

for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=10,
            fontweight='bold', color='#222222')

y_top = max(rates) * 1.18
ax.plot([0, 0, 1, 1],
        [y_top - 0.12, y_top, y_top, y_top - 0.12],
        color='#444444', linewidth=0.8, solid_capstyle='round')

_, p_main = stats.mannwhitneyu(
    low_all['went_extinct'].astype(int),
    high_all['went_extinct'].astype(int),
    alternative='two-sided'
)
ax.text(0.5, y_top + 0.04,
        f'$p = 1.8 \\times 10^{{-274}}$',
        ha='center', va='bottom', fontsize=8, color='#444444')

rel_main = (rates[0] - rates[1]) / rates[0]
ax.text(0.5, max(rates) * 0.82,
        f'{rel_main:.1%} reduction',
        ha='center', va='center', fontsize=8,
        color='#666666', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(
    ['Low $\\mathit{\\sigma^2_L}$', 'High $\\mathit{\\sigma^2_L}$'],
    fontsize=10)
ax.set_ylabel('Local extinction rate (%)', fontsize=9, labelpad=6)
ax.set_ylim(0, y_top * 1.25)
ax.set_xlim(-0.55, 1.55)
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.tick_params(axis='both', which='both', length=3)
ax.set_title('BBS prospective extinction', fontsize=9,
             fontweight='bold', pad=8)
ax.text(0.5, 1.02,
        '$n = 198{,}557$ pairs · 3,877 routes · 569 species',
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=7.5, color='#666666')
ax.text(-0.12, 1.06, 'a', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

ax = axes[1]

reductions = [rel_low * 100, rel_high * 100]
pvals_str  = [f'$p = 7.9 \\times 10^{{-18}}$',
              f'$p = 1.0 \\times 10^{{-4}}$']
colors2    = ['#2E86AB', '#6A4C93']
x2         = [0, 1]

bars2 = ax.bar(x2, reductions, color=colors2, width=0.45,
               edgecolor='none', zorder=3)

for bar, red, pval in zip(bars2, reductions, pvals_str):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f'{red:.1f}%',
            ha='center', va='bottom', fontsize=10,
            fontweight='bold', color='#222222')
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            pval,
            ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold')

ax.annotate('',
            xy=(1.225, reductions[1]),
            xytext=(-0.225, reductions[0]),
            arrowprops=dict(
                arrowstyle='->', color='#AAAAAA',
                lw=1.0, connectionstyle='arc3,rad=-0.15'))
ax.text(0.5, (reductions[0] + reductions[1]) / 2 + 3,
        'Predicted\nattenuation (P4)',
        ha='center', va='center', fontsize=7.5,
        color='#AAAAAA', style='italic')

# x-axis labels
ax.set_xticks(x2)
ax.set_xticklabels(['Low $c_1$', 'High $c_1$'], fontsize=10)
for tick_x, sub in zip(x2, ['(env. dominated)', '(demo. dominated)']):
    ax.text(tick_x, -7, sub, ha='center', va='top',
            fontsize=7.5, color='#666666', transform=ax.transData)

low_sp  = ('Song Sparrow · Red-eyed Vireo\n'
           'Common Yellowthroat · Ovenbird')
high_sp = ('Barn Swallow · Red-winged Blackbird\n'
           'House Sparrow · Chimney Swift')
ax.text(0, -15, low_sp, ha='center', va='top', fontsize=7,
        color='#2E86AB', style='italic', linespacing=1.5)
ax.text(1, -15, high_sp, ha='center', va='top', fontsize=7,
        color='#6A4C93', style='italic', linespacing=1.5)

ax.set_ylabel('Relative reduction in\nextinction rate (%)',
              fontsize=9, labelpad=6)
ax.set_ylim(0, 80)
ax.set_xlim(-0.55, 1.55)
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.tick_params(axis='both', which='both', length=3)
ax.set_title('$c_1$ modulation within BBS', fontsize=9,
             fontweight='bold', pad=8)
ax.text(0.5, 1.02,
        '$n = 34{,}010$ pairs per quartile',
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=7.5, color='#666666')
ax.text(-0.18, 1.06, 'b', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top')

png_path = os.path.join(OUT_DIR, 'fig5.png')
pdf_path = os.path.join(OUT_DIR, 'fig5.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved {png_path}")
print(f"Saved {pdf_path}")
