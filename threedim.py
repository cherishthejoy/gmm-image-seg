import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GM
from sklearn.preprocessing import StandardScaler

img = Image.open("images/bad-red.png")

# This just in case if the image is png
# and has the alpha channel intact
img = img.convert('RGB') 

img2 = np.array(img)
rows, cols, ch = img2.shape

img3 = img2.reshape((-1, 3))

scaler = StandardScaler()
img34 = scaler.fit_transform(img3)

gmm = GM(n_components=3, covariance_type='spherical').fit(img34)
labels = gmm.predict(img34)

cmap = plt.get_cmap("magma")
colored = cmap(labels / labels.max())[:, :3]

segmented_img = (colored * 255).astype(np.uint8).reshape(rows, cols, 3)
final = Image.fromarray(segmented_img)

idx = np.random.choice(len(img3), size = 5000, replace=False)
sample = img3[idx]
sample_labels = labels[idx]


plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.imshow(img)

ax = fig.add_subplot(132, projection='3d')
ax.scatter(sample[:,0], sample[:,1], sample[:,2], c=sample_labels, s=5, alpha=0.3)

plt.subplot(133)
plt.imshow(final)

plt.show()



