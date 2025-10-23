import numpy as np
import insightface
import cv2

model = insightface.app.FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0)

def extract_embedding(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = model.get(img)
    if not faces:
        raise Exception("No face detected")
    return faces[0].normed_embedding

def compare_embeddings(emb1, emb2):
    # Both embeddings assumed normalized
    return np.dot(emb1, emb2)
