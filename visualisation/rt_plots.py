#%%
import matplotlib.pyplot as plt
import scienceplots

plt.rcParams["text.usetex"] = True
plt.style.use('science')


fig, (ax, bx) = plt.subplots(1, 2,  tight_layout=True, dpi=300, figsize=(6.4, 3))

im1 = plt.imread('office_reconstruction.png')
im2 = plt.imread('office_reconstruction_ga.png')

ax.imshow(im1)
ax.axis('off')
bx.imshow(im2)
bx.axis('off')
# ax.set_title('Grid A')
# bx.set_title('Grid B')

plt.show()
# %%
fig.savefig('rt-room-test.png', transparent=True)


# %%
