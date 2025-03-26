from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define class labels and counts
class_labels = np.array(["class1", "class7", "class4", "class2",
                         "class8", "class6", "class10", "class9", "class3", "class5"])
class_counts = np.array([21243, 12174, 10333, 10246, 9269, 9089, 6847, 5959, 4576, 4076])



# Compute class weights
total_samples = sum(class_counts)
num_classes = len(class_counts)
class_weights = {label: total_samples / (num_classes * count)
                 for label, count in zip(class_labels, class_counts)}

print(class_weights)
