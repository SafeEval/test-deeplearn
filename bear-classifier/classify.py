import os
import sys

from fastbook import *
from fastai.vision.widgets import *
import PIL

if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} <image path>')
    sys.exit(1)

learn_inf = load_learner('model-export.pkl')
print(f'Learner vocab: {learn_inf.dls.vocab}')

img_path = Path(sys.argv[1])
img = PIL.Image.open(img_path).convert('RGB')


pred,pred_idx,probs = learn_inf.predict(img_path)
result = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
print(result)
