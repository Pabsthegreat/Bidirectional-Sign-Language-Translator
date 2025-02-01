import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your labels
labels = np.load("sign_language_dataset/labels.npy")

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Check if all labels are properly encoded
print("Unique Original Labels:", np.unique(labels))
print("Unique Encoded Labels:", np.unique(encoded_labels))

# Create a mapping of original labels to encoded values
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Print label mapping
print("\nLabel Encoding Mapping:")
for label, encoded in label_mapping.items():
    print(f"{label}: {encoded}")

# Decode back to original to verify correctness
decoded_labels = label_encoder.inverse_transform(encoded_labels)

# Check if decoded labels match original labels
if np.array_equal(labels, decoded_labels):
    print("\nLabel encoding and decoding is correct!")
else:
    print("\nError: Mismatch found in encoding/decoding!")
