class LeafEdgeDetection():
  def __init__(self):
    import os
    import cv2
    def edge_detect(folder):
      for image in os.listdir(folder):
        path = os.path.join(folder, image)
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        cv2.imwrite(path, edges)
    edge_detect('leafdata/Training')
    edge_detect('leafdata/Validation')
    edge_detect('leafdata/Testing')
    edge_detect('leafdata/Predict')
