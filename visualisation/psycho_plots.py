# %%
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import scienceplots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score

plt.rcParams["text.usetex"] = True
plt.style.use('science')

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# %%
t = np.arange(0, 4, 0.01)
a = np.sin(2 * np.pi * t)

pos = [0, 45, 90, 135, 180, 225, 270, 315]
dis = np.random.uniform(low=0.3, high=1, size=len(pos))
head = plt.imread('listener_head_topview.png')
head = OffsetImage(head, zoom=0.2)
ab = AnnotationBbox(head, (0, 0), xycoords='data', frameon=False)

fig, ax = plt.subplots(
    subplot_kw={'projection': 'polar'},
    figsize=(4, 4),
    dpi=300,
    tight_layout=True)
ax.plot(np.radians(pos), dis, 'ro')
ax.add_artist(ab)
ax.set_label('')

plt.show()

# %%
fig, ax = plt.subplots(
    subplot_kw={'projection': 'polar'},
    figsize=(4, 4),
    dpi=300,
    tight_layout=True)
# %%
localisation_df = pd.read_excel('filtered_loc.xlsx')

#%%
rt_df = localisation_df.loc[(localisation_df['RT'] == True)]
nort_df = localisation_df.loc[(localisation_df['RT'] == False)]

error_near    = nort_df.loc[(localisation_df['ArrayDistance'] == 'Near')]
error_mid     = nort_df.loc[(localisation_df['ArrayDistance'] == 'Medium')]
error_far     = nort_df.loc[(localisation_df['ArrayDistance'] == 'Far')]
error_near_rt = rt_df.loc[(localisation_df['ArrayDistance'] == 'Near')]
error_mid_rt  = rt_df.loc[(localisation_df['ArrayDistance'] == 'Medium')]
error_far_rt  = rt_df.loc[(localisation_df['ArrayDistance'] == 'Far')]

#%% PSPP export



# %%

ax_title_fs = 14
label_fontsize = 12
# Thesis
fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 3.4)) 
# Ieee

ticks = [i * 5 for i in range(7)]
ticklabels = ['${}^\circ$'.format(i) for i in ticks]

ax.boxplot([error_near['AngularDistance'], error_mid['AngularDistance'], error_far['AngularDistance']],
           notch=False, showmeans=True, labels=['Near', 'Medium', 'Far'],
           flierprops=dict(markerfacecolor='g', marker='D'), widths=0.5, showfliers=False)

ax.set_title('Anechoic Set (HRTF)', fontsize=ax_title_fs)
ax.set_yticks(ticks)
ax.tick_params(axis='x', which='minor', length=0)


bx.boxplot([error_near_rt['AngularDistance'], error_mid_rt['AngularDistance'], error_far_rt['AngularDistance']],
           notch=False, showmeans=True, labels=['Near', 'Medium', 'Far'],
           flierprops=dict(markerfacecolor='g', marker='D'), widths=0.5, showfliers=False)
bx.set_title('Pipeline Set (HRTF + BRIR)', fontsize=ax_title_fs)
bx.set_yticks(ticks)
bx.set_yticklabels(ticklabels)
bx.tick_params(axis='x', which='minor', length=0)

ax.set_ylim(0, 32)
bx.set_ylim(0, 32)
# fig.suptitle('Localisation Error', fontsize=14, y=0.93)
plt.show()

# %%
fig.savefig('localisation_accuracy_ieee.pdf', transparent=True)
# %%

fig, (ax, bx, cx) = plt.subplots(1, 3, tight_layout=True, dpi=300, figsize=(6.4, 2.3))

ax.hist([error_near['AngularDistance'], error_near_rt['AngularDistance']],
        bins=6)
ax.set_title('Near')

bx.hist([error_mid['AngularDistance'], error_mid_rt['AngularDistance']], bins=6)
bx.set_title('Mid')
bx.legend(['Anechoic', 'Pipeline'])

cx.hist([error_far['AngularDistance'], error_far_rt['AngularDistance']], bins=6)
cx.set_title('Far')

for axes in (ax, bx, cx):
    ticks = [50, 100, 150]
    axes.set_xticks(ticks, ['${}^\circ$'.format(i) for i in ticks])

# fig.suptitle('Localisation Error Distributions', y=0.92, fontsize=14)
plt.show()

# %%
fig.savefig('localisation_distributions_ieee.pdf', transparent=True)
# %%

masking_df = pd.read_excel('masking_filtered.xlsx')
nort = masking_df.loc[masking_df['RT'] == False]
rt = masking_df.loc[masking_df['RT'] == True]
# %%


# TODO: identify success rate, angular threshold single to multiple

# %%


# Assuming y_true contains true labels (0 or 1) and y_scores contains predicted probabilities or scores

# fig, ax = plt.subplots(1, tight_layout=True, dpi=300, figsize=(6.4, 3))
fig, ax = plt.subplots(1, tight_layout=True, dpi=300, figsize=(4.6, 3))

fpr, tpr, thresholds = roc_curve(nort['TrueValue'].tolist(), nort['BinaryResponse'].tolist())
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, lw=2, label=r'ROC \textbf{Anechoic} ' + '(AUC = {:.2f})'.format(roc_auc))
ax.plot([0, 1], [0, 1], lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')



fpr, tpr, thresholds = roc_curve(rt['TrueValue'].tolist(), rt['BinaryResponse'].tolist())
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, lw=2, label=r'ROC \textbf{Pipeline} ' + '(AUC = {:.2f})'.format(roc_auc))

ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')

ax.legend(loc='lower right')

ax.set_title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# %%

# fig.savefig('masking_roc.pdf', transparent=True)
fig.savefig('masking_roc_ieee.pdf', transparent=True)
# %%

angles = masking_df['Angle'].unique()
print('Angle, $F_1$ Anechoic, $F_1$ Pipeline, $F_1$ Aggregate')

for angle in sorted(angles, reverse=True):
    y_pred_rt = rt.loc[abs(masking_df['Angle'] - angle) < 0.1]['BinaryResponse'].tolist()
    y_true_rt = rt.loc[abs(masking_df['Angle'] - angle) < 0.1]['TrueValue'].tolist()

    y_pred_nort = nort.loc[abs(masking_df['Angle'] - angle) < 0.1]['BinaryResponse'].tolist()
    y_true_nort = nort.loc[abs(masking_df['Angle'] - angle) < 0.1]['TrueValue'].tolist()

    y_pred_agg = masking_df.loc[abs(masking_df['Angle'] - angle) < 0.1]['BinaryResponse'].tolist()
    y_true_agg = masking_df.loc[abs(masking_df['Angle'] - angle) < 0.1]['TrueValue'].tolist()

    # print('{:3d}, {:3.2}, {:3.2}, {:3.2}, {:3.2}, {:3.2}, {:3.2}, '.format(
    #       round(angle), 
    #       f1_score(y_true_rt, y_pred_rt, average='micro'),
    #       f1_score(y_true_nort, y_pred_nort, average='micro')),
    #       f1_score(y_true_agg, y_pred_agg, average='micro'),
          
    #       accuracy_score(y_true_rt, y_pred_rt),
    #       accuracy_score(y_true_nort, y_pred_nort),
    #       accuracy_score(y_true_agg, y_pred_agg)
    #       )
    
    print('${:3d}^\\circ$, {:3.2}, {:3.2}, {:3.2} '.format(
          round(angle), 
          f1_score(y_true_nort, y_pred_nort, average='micro'),
          f1_score(y_true_rt, y_pred_rt, average='micro'),
          f1_score(y_true_agg, y_pred_agg, average='micro'))
          )
    
    # print(
    #       angle, 
    #       f1_score(y_true_rt, y_pred_rt, average='micro'),
    #       f1_score(y_true_nort, y_pred_nort, average='micro'))

# %%
