import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture as GMM

img = np.array(Image.open("fruits.jpg"))
img2 = img.reshape((-1, 3))

gmm = GMM(n_components=3, covariance_type='tied').fit(img2)
gmm_labels = gmm.predict(img2)

or_shape = img.shape
segmented = gmm_labels.reshape(or_shape[0], or_shape[1])

mask = (segmented * 255).astype(np.uint8)
Image.fromarray(mask).save("gmm_segmented_mask.png")

# Take a random sample (to avoid plotting millions of points)
idx = np.random.choice(len(img2), size=1000, replace=False)
sample = img2[idx]
sample_labels = gmm_labels[idx]

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(sample[:,0], sample[:,1], sample[:,2], c=sample_labels, s=5, alpha=0.3)
ax1.set_xlabel("R")
ax1.set_ylabel("G")
ax1.set_zlabel("B")
ax1.set_title("GMM color clusters")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(sample[:,0], sample[:,1], sample[:,2], c=sample_labels, s=5, alpha=0.3)

plt.show()



