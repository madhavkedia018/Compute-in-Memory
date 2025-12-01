import Overlay_module
print(Overlay_module.available_params(Overlay_module.NETWORK_CNVW1A1))

bnn_classifier = Overlay_module.CnvClassifier(Overlay_module.NETWORK_CNVW1A1,'cifar10',Overlay_module.RUNTIME_BNN)
cnn_classifier = Overlay_module.CnvClassifier(Overlay_module.NETWORK_CNVW1A1,'cifar10',Overlay_module.RUNTIME_CNN)
print(bnn_classifier.classes)

from PIL import Image

def load_for_bnn(path, display_img=True):
    im = Image.open(path)
    if display_img:
        display(im)
    im.load()                    
    im = im.convert("RGB")       
    im = im.resize((32, 32))     
    return im


from PIL import Image
import numpy as np

im = load_for_bnn('images/cat.jpg')

class_out=bnn_classifier.classify_image(im)
print("Class number: {0}".format(class_out))
print("Class name: {0}".format(bnn_classifier.class_name(class_out)))

class_out=cnn_classifier.classify_image(im)
print("Class number: {0}".format(class_out))
print("Class name: {0}".format(cnn_classifier.class_name(class_out)))





%matplotlib inline
import matplotlib.pyplot as plt

img_class = bnn_classifier.classify_image_details(im)
print(img_class)

x_pos = np.arange(len(img_class))
fig, ax = plt.subplots()
ax.bar(x_pos, (img_class/100.0), 0.25)
ax.set_xticklabels(bnn_classifier.classes, rotation='vertical')
ax.set_xticks(x_pos)
ax.set
plt.title(bnn_classifier.class_name(class_out))
plt.show()






import numpy as np
from PIL import Image

# Load images
im1 = load_for_bnn('images/horse.jpg')
im2 = load_for_bnn('images/deer.jpg')
im3 = load_for_bnn('images/airplane.jpg')
im4 = load_for_bnn('images/cat.jpg')

class_out_1=bnn_classifier.classify_image(im1)
label_1 = bnn_classifier.class_name(class_out_1)
print(label_1)

class_out_2=bnn_classifier.classify_image(im2)
label_2 = bnn_classifier.class_name(class_out_2)
print(label_2)

class_out_3=bnn_classifier.classify_image(im3)
label_3 = bnn_classifier.class_name(class_out_3)
print(label_3)

class_out_4=bnn_classifier.classify_image(im4)
label_4 = bnn_classifier.class_name(class_out_4)
print(label_4)

img_class_1 = bnn_classifier.classify_image_details(im1)
img_class_2 = bnn_classifier.classify_image_details(im2)
img_class_3 = bnn_classifier.classify_image_details(im3)
img_class_4 = bnn_classifier.classify_image_details(im4)

all_outputs = np.vstack([
    img_class_1, img_class_2, img_class_3, img_class_4
]) / 100.0

labels = [label_1, label_2, label_3, label_4]

num_models = all_outputs.shape[0]   
num_classes = all_outputs.shape[1]  

x = np.arange(num_classes)          
bar_width = 0.2                     

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(num_models):
    ax.bar(
        x + i * bar_width,
        all_outputs[i],
        width=bar_width,
        label=labels[i]     
    )

ax.set_xticks(x + bar_width * (num_models - 1) / 2)
ax.set_xticklabels(bnn_classifier.classes, rotation=45, ha='right')

ax.set_ylabel("Probability")
ax.set_title("Comparison of Class Probabilities for 4 Outputs")
ax.legend()

plt.tight_layout()
plt.show()


