#
from ezfaces.face_classifier import faceClassifier
import cv2


fc = faceClassifier()
lbl_new = fc.add_img_data(from_webcam=True)
fc.train()
# take a snapshot from webcam
x_novel = fc.webcam2vec()
x_pred, lbl_pred = fc.classify(x_novel)
print("The ID of the newly added subject is %d. The prediction from "
        "the webcam is %d" %(lbl_new, lbl_pred))
cv2.imshow("Prediction", fc.vec2img(x_pred))
cv2.waitKey(3000)
cv2.destroyAllWindows()
