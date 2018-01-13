import cv2
import sys
import pickle
import numpy as np
from scipy.stats import norm


class PhoneFinder:
    def __init__(self, base_threshold=60, threshold_step=10, kernel_size=8):
        self.base_thresh = base_threshold
        self.thresh_step = threshold_step
        self.k_size = kernel_size
        self.prediction = np.empty(0)
        self.model = None

    def fit(self, training_data, labels):
        aspect_ratios = np.empty((len(training_data), 1))
        for i, sample in enumerate(training_data):
            img = cv2.imread(training_dir+sample, cv2.IMREAD_GRAYSCALE)
            phone_contour = self._find_phone_contour_from_label(img, labels[i])
            aspect_ratios[i] = phone_contour.get_aspect_ratio()

        self.model = norm(aspect_ratios.mean(), aspect_ratios.std())

    def predict(self, test_data):
        self.prediction = np.empty((len(test_data), 2))
        for i, sample in enumerate(test_data):
            img = cv2.imread(training_dir+sample, cv2.IMREAD_GRAYSCALE)
            for th in range(self.base_thresh, 256, self.thresh_step):
                contours = [Contour(p) for p in self._get_image_contours(img, th)]
                if len(contours) > 0:
                    break

            aspect_ratios = np.array([c.get_aspect_ratio() for c in contours])

            phone_score = self.model.pdf(aspect_ratios)
            guess = contours[phone_score.argmax()]
            guess.scale_center(*img.shape)
            self.prediction[i] = guess.center

    def _find_phone_contour_from_label(self, img, label):
        for th in range(self.base_thresh, 256, self.thresh_step):
            contours = [Contour(p) for p in self._get_image_contours(img, thresh=th)]
            if len(contours) == 0:
                continue

            center_xy = (img.shape[1]*label[0], img.shape[0]*label[1])
            for contour in contours:
                if contour.contains_point(center_xy):
                    return contour
        return None

    def _get_image_contours(self, img, thresh=60):
        kernel = np.ones((self.k_size, self.k_size), np.uint8)
        _, bin_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
        morph = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        return cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]


class Contour:
    def __init__(self, points):
        self.points = points
        self.center = self._compute_center()

    def scale_center(self, m, n):
        self.center = (self.center[0] / n, self.center[1] / m)

    def contains_point(self, point):
        return cv2.pointPolygonTest(self.points, point, False) == 1

    def get_aspect_ratio(self):
        rect = cv2.minAreaRect(self.points)[1]
        if rect[0] == 0 or rect[1] == 0:
            return 0
        return rect[0] / rect[1] if rect[0] > rect[1] else rect[1] / rect[0]

    def _compute_center(self):
        M = cv2.moments(self.points)
        cx = M['m10'] / (M['m00'] + 1e-6)  # To prevent division by zero
        cy = M['m01'] / (M['m00'] + 1e-6)
        return cx, cy


def load_training_data(dir_path="./find_phone/"):
    samples = np.empty(0)
    labels = np.empty((0, 2))
    with open(dir_path+"labels.txt") as f:
        line = f.readline()
        while line:
            label = line.split()
            samples = np.append(samples, label[0])
            labels = np.vstack((labels, (float(label[1]), float(label[2]))))
            line = f.readline()
    return samples, labels


def calculate_error(prediction, labels):
        score = np.all(abs(prediction - labels) < 0.05, axis=1).sum(dtype='float')
        return score / len(prediction)


if __name__ == "__main__":
    training_dir = "./find_phone/"  # sys.argv[1]
    train_data, labels = load_training_data(training_dir)

    phone_finder = PhoneFinder()
    phone_finder.fit(train_data, labels)
    pickle.dump(phone_finder, open("phone_finder.p", "wb"))

    # phone_finder.predict(samples[valid_index])

    # score = calculate_error(phone_finder.prediction, labels[valid_index])
    # print("Error: {}".format(1 - score.mean()))
