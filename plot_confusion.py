from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Only use the labels that appear in the data
cmap=plt.cm.Blues
classes = ['NC', 'MCI']
cm = np.asarray([[0.9143,0.0857],[0.0857 ,0.9143]])
print(type(cm))
print(cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       # ... and label them with the respective list entries
       xticklabels=classes, yticklabels=classes,
       title='normalized confusion matrix',
       ylabel='True label',
       xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",size=20,
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
 ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(13)

plt.savefig('data/confusion.png',dpi=600)