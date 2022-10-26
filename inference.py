import darknet
import cv2
import os
import sys


class DarknetYoloModel:

    def __init__(self, cfg_path, obj_data_path, weights_path) -> None:
        network_info = darknet.load_network(cfg_path, obj_data_path, weights_path)
        self.network, self.class_names, self.class_colors = network_info
        self.input_width = darknet.network_width(self.network)
        self.input_height = darknet.network_height(self.network)

    def detect(self, image):
        darknet_image = darknet.make_image(self.input_width,  self.input_height, 3)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        img_height, img_width, _ = image.shape
        width_ratio = img_width / self.input_width
        height_ratio = img_height / self.input_height
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

        detections = darknet.detect_image(self.network, self.class_names, darknet_image)

        darknet.free_image(darknet_image)

        return detections, width_ratio, height_ratio

def main():
    data_file_path = sys.argv[1]
    cfg_path = sys.argv[2]
    obj_data_path = sys.argv[3]
    weights_path = sys.argv[4]

    model = DarknetYoloModel(cfg_path, obj_data_path, weights_path)

    image_paths = []
    with open(data_file_path, 'r') as data_file:
        image_paths = data_file.readlines()

    index = 0
    while True:
        image = cv2.imread(image_paths[index].replace('\n', ''))

        detections, width_ratio, height_ratio = model.detect(image)

        for label, confidence, bbox in detections:
            left, top, right, bottom = darknet.bbox2points(bbox)
            left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(
                bottom * height_ratio)
            cv2.rectangle(image, (left, top), (right, bottom), model.class_colors[label], 2)
            cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        model.class_colors[label], 2)
        cv2.imshow('ex', image)
        key = cv2.waitKey()
        if key == ord('d'):
            index = min(index + 1, len(image_paths) - 1)
        elif key == ord('a'):
            index = max(index - 1, 0)
        elif key == ord('q'):
            break
        else:
            pass




if __name__ == "__main__":
   main()