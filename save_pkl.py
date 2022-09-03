import os
import pickle

from google.cloud import storage

filename = ".pkl"

# Save model to local filesystem
with open(filename, 'wb') as f:
    pickle.dump(model, f)

# Upload model to Cloud Storage
blob = storage.blob.Blob.from_string(filename, client = storage.Client())
blob.upload_from_filename(filename)