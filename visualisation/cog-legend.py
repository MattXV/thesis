#%%
import scienceplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams["text.usetex"] = True
plt.style.use('science')

#%%

# LOW Alpha
row1 = {
    "Air": [104,51,17],
    "Glass": [254,4,107],
    "Masonry": [91,126,238],
    "Studwork": [60,210,154],
    "Wood": [46,176,54],
    "Floor": [196,79,10],
    "Panelling": [198,65,246],
    "Treatment": [212,193,129],
    "Audience": [7,160,32],
    "Ceilings": [21,181,36],
    "Wool": [139,27,47],
    "Other": [83,223,164],
}

row2 = {
    "ph": [255,255,255],
    "Glass": [71,159,57],
    "Masonry": [123,204,27],
    "Studwork": [188,10,227],
    "Wood": [170,202,209],
    "Floor": [6,251,120],
    "Panelling": [218,105,114],
    "Treatment": [57,201,71],
    "Audience": [125,217,39],
    "Ceilings": [146,243,247],
    "Wool": [186,159,242],
    "Other": [229,56,43],
}


width = 10
fig, ax = plt.subplots(figsize=(width, 1))


rect_width = width / len(row1)
rect_height = 0.3
fontsize = 12
y_pos = 0
for i in range(len(row1)):
    x_pos = (i / width) * rect_width
  
    label = list(row1.keys())[i]
    colour = np.array(row1[label], dtype=np.float32) / 255

    r = Rectangle((x_pos, y_pos), rect_width, rect_height, color=colour)
    t = plt.text(x_pos, rect_height + 0.1 + y_pos, label, color=colour, rotation=0, fontsize=fontsize)
    ax.add_artist(r)
    ax.add_artist(t)

plt.show()
ax.set_axis_off()
fig.set_layout_engine('tight')
fig.savefig('cog-legend-top.pdf')

fig, ax = plt.subplots(figsize=(width, 1))

y_pos = 0
for i in range(len(row2)):
    x_pos = (i / width) * rect_width
  
    label = list(row2.keys())[i]
    colour = np.array(row2[label], dtype=np.float32) / 255

    r = Rectangle((x_pos, y_pos), rect_width, rect_height, color=colour)
    t = plt.text(x_pos, rect_height + 0.1 + y_pos, label, color=colour, rotation=0, fontsize=fontsize)
    ax.add_artist(r)
    ax.add_artist(t)

ax.set_axis_off()
fig.set_layout_engine('tight')
fig.savefig('cog-legend-bot.pdf')

plt.show()
# %% CST VIS


materials = {
  "Generic":             [0.1215686, 0.4666667, 0.7058824],
  "Brick":               [0.6823529, 0.7803922, 0.9098039],
  "Carpet":              [1.0000000, 0.4980392, 0.0549020],
  "Ceramic":             [0.1725490, 0.6274510, 0.1725490],
  "Fabric":              [0.5960784, 0.8745098, 0.5411765],
  "Glass":               [1.0000000, 0.5960784, 0.5882353],
  "Leather":             [0.5803922, 0.4039216, 0.7411765],
  "Metal":               [0.5490196, 0.3372549, 0.2941176],
  "Painted":             [0.7686275, 0.6117647, 0.5803922],
  "Plastic":             [0.8901961, 0.4666667, 0.7607843],
  "Stone":               [0.4980392, 0.4980392, 0.4980392],
#   "Stone_polished":      [0.7803922, 0.7803922, 0.7803922],
  "Tile":                [0.8588235, 0.8588235, 0.5529412],
  "Wallpaper":           [0.0901961, 0.7450980, 0.8117647],
  "Wood":                [0.6196078, 0.8549020, 0.8980392]
}

width = 10
fig, ax = plt.subplots(figsize=(width, 1))


rect_width = width / len(materials)
rect_height = 0.3
fontsize = 12
y_pos = 0
j = -0.3
for i in range(len(materials)):
    x_pos = (i / width) * rect_width
  
    label = list(materials.keys())[i]
    colour = np.array(materials[label], dtype=np.float32)

    offset = j if j > 0 else -0.4

    r = Rectangle((x_pos, y_pos + 0.3), rect_width, rect_height, color=colour)
    t = plt.text(x_pos, rect_height + 0.1 + y_pos + offset, label, color=colour, rotation=0, fontsize=fontsize)

    j *= -1

    ax.add_artist(r)
    ax.add_artist(t)

ax.set_axis_off()
fig.set_layout_engine('tight')
plt.show()
fig.savefig('cst-mat-vis.pdf')
# %%
