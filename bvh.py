import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.style.use('science')


fig, (ax, bx, cx) = plt.subplots(1, 3,  tight_layout=True, dpi=300, figsize=(6.4, 3))

im1 = plt.imread('bvh/shaded-geometry.png')
im2 = plt.imread('bvh/wireframe-geometry.png')
im3 = plt.imread('bvh/bvh.png')

ax.imshow(im1)
bx.imshow(im2)
cx.imshow(im3)

plt.show()