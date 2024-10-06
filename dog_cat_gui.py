import sys
import torch
from torchvision import transforms
from torch import nn
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QPushButton, QFileDialog
from PIL import Image as PILImage



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 clase (pisică și câine)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = CNN()
model.load_state_dict(torch.load('cat_dog_classifier.pth', weights_only=True))
model.eval()

def predict_image(image_path):
    image = PILImage.open(image_path)
    print(f"Imagine încărcată: {image_path}")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    prediction_result = "Câine" if predicted.item() == 1 else "Pisică"
    print(f"Predicție: {prediction_result}")
    return prediction_result


class App(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Detector Pisică/Câine')
        self.setGeometry(100, 100, 400, 300)
        self.setStyleSheet("background-color: lightblue;")

        self.layout = QtWidgets.QVBoxLayout()

        self.btn_load = QPushButton('Încarcă Imagine', self)
        self.btn_load.clicked.connect(self.load_image)
        self.layout.addWidget(self.btn_load)

        self.img_label = QLabel(self)
        self.layout.addWidget(self.img_label)

        self.result_label = QLabel("Predicție: ", self)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Alege o imagine', '', 'Imagini (*.png *.jpg *.jpeg)')
        if file_path:

            image = QtGui.QImage(file_path)
            image = image.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
            self.img_label.setPixmap(QtGui.QPixmap(image))


            result = predict_image(file_path)
            self.result_label.setText(f"Predicție: {result}")
            print(f"Imagine încărcată: {file_path}")
            print(f"Predicție: {result}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
