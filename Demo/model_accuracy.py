import Overlay_module

labels = []
with open("/home/xilinx/jupyter_notebooks/Major_Project/cifar-10-batches-bin/test_batch.bin", "rb") as file:
    #for 10000 pictures
    for i in range(10000):
        #read first byte -> label
        labels.append(int.from_bytes(file.read(1), byteorder="big"))
        #read image (3072 bytes) and do nothing with it
        file.read(3072)
    file.close()


bnn_classifier = Overlay_module.CnvClassifier(Overlay_module.NETWORK_CNVW1A1,'cifar10',Overlay_module.RUNTIME_BNN)
result_bnn = bnn_classifier.classify_cifars("/home/xilinx/jupyter_notebooks/Major_Project/cifar-10-batches-bin/test_batch.bin")
time_bnn = bnn_classifier.usecPerImage

countRight = 0
for idx in range(len(labels)):
    if labels[idx] == result_bnn[idx]:
        countRight += 1
accuracybnn = countRight*100/len(labels)

print(accuracybnn)





import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

# CIFAR-10 class names
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# Convert to numpy arrays
labels_np = np.array(labels)
pred_np   = np.array(result_bnn)

# -----------------------------
# Build confusion matrix manually
# -----------------------------
num_classes = 10
cm = np.zeros((num_classes, num_classes), dtype=int)

for t, p in zip(labels_np, pred_np):
    cm[t, p] += 1

print("Confusion Matrix (raw values):")
print(cm)

# -----------------------------
# Plot confusion matrix
# -----------------------------
plt.figure(figsize=(10,7))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("BNN Confusion Matrix (CIFAR-10)")
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

