import sys
import os
from os.path import join as opj
from os.path import dirname as opd

import tifffile
import numpy as np
import time
import traceback

from omegaconf import OmegaConf
from PyQt6.QtCore import QUrl
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QStandardItemModel, QStandardItem, QPixmap, QImage, QDoubleValidator, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QSlider,
    QLineEdit,
    QProgressBar,
    QMessageBox,
    QCheckBox,
)
from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QIcon
import torch
import subprocess

from GUI_Utils.Utils import load_tiff_seq, ReconstructThread


class VideoReconstructionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MUFFLE GUI")
        # 设置图标
        self.setWindowIcon(QIcon("./GUI_Utils/icon.png"))

        self.output_folder = "./outputs"
        self.ch_a_numpy = None
        self.ch_b_numpy = None
        self.linear_gain = 20
        self.match_output = False
        self.enhance_a = True
        self.enhance_b = True

        self.opt_path = "./GUI_Utils/Model/Fn15_Model.yaml"
        self.model_path = "./GUI_Utils/Model/Fn15_Model.pkl"

        print("CUDA Device Count: " + str(torch.cuda.device_count()))
        self.device_list = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + ["cpu"]
        self.selected_device = self.device_list[0]

        self.create_widgets()
        self.create_layout()

        try:
            self.ch_a_numpy = self.loadAndSetInputPath("./GUI_Utils/Test_Img/S4101_0000_Test/acceptor_short.tif", "a")
            self.ch_b_numpy = self.loadAndSetInputPath("./GUI_Utils/Test_Img/S4101_0000_Test/donor_short.tif", "b")
        except:
            self.ch_a_numpy = np.ones([50, 200, 200]) * 0
            self.ch_a_numpy = np.ones([50, 200, 200]) * 0
            print(f"Preload test images failed! Please manually load the input images.")
        self.ch_a_out_numpy = np.ones_like(self.ch_a_numpy) * 0
        self.ch_b_out_numpy = np.ones_like(self.ch_b_numpy) * 0

        if len(self.device_list) == 1 and self.device_list[0] == "cpu":
            # 弹出提示框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Warning!")
            msg_box.setText(
                f"GPU is not available! Please check your CUDA installation if you have a GPU. \nMUFFLE will run on CPU. The reconstruction will be extremely slow. \nWe recommend using GPU."
            )
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()

    def create_widgets(self):
        def create_video_viewer(name, size):
            video_viewer = QLabel(name)
            # video_viewer.setPixmap(QPixmap("GUI_Utils/test.png"))
            video_viewer.setFixedSize(*size)
            video_viewer.setAcceptDrops(True)
            video_viewer.installEventFilter(self)
            # 设置图像居中
            video_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            video_viewer.setScaledContents(True)
            return video_viewer

        def create_view_slider(name, size, value_changed_func):
            view_slider = QSlider(Qt.Orientation.Horizontal, self)
            view_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            view_slider.setFixedSize(*size)
            view_slider.valueChanged[int].connect(value_changed_func)
            return view_slider

        bfont = QFont()
        bfont.setBold(True)

        self.input_label1 = QLabel("Acceptor dark")
        self.input_label1.setFont(bfont)
        self.input_path1 = QLabel("Path: None")
        self.input_path1.setWordWrap(True)
        self.input_path1.setMaximumHeight(50)
        self.input_label2 = QLabel("Donor dark")
        self.input_label2.setFont(bfont)
        self.input_path2 = QLabel("Path: None")
        self.input_path2.setWordWrap(True)
        self.input_path2.setMaximumHeight(50)
        self.output_label1 = QLabel("Acceptor recon.")
        self.output_label1.setFont(bfont)
        self.output_label2 = QLabel("Donor recon.")
        self.output_label2.setFont(bfont)

        self.input_video1 = create_video_viewer("Acceptor dark", (300, 300))
        self.in1_sld = create_view_slider("Acceptor dark", (300, 20), self.changeValueIn1)

        self.input_video2 = create_video_viewer("Donor dark", (300, 300))
        self.in2_sld = create_view_slider("Donor dark", (300, 20), self.changeValueIn2)

        self.output_video1 = create_video_viewer("Acceptor recon.", (300, 300))
        self.out1_sld = create_view_slider("Acceptor recon.", (300, 20), self.changeValueOut1)

        self.output_video2 = create_video_viewer("Donor recon.", (300, 300))
        self.out2_sld = create_view_slider("Donor recon.", (300, 20), self.changeValueOut2)

        self.instructions_label = QLabel(
            'Instructions: \n1. Drag and drop the input videos to the video viewers. \n2. Set the linear gain. \n3. Drag and drop the output path to the output area. \n4. Select the device to use. \n5. Click the "Reconstruct with MUFFLE" button. \n6. The enhanced videos will be saved to output folder.'
        )
        self.instructions_label.setWordWrap(True)

        self.output_label = QLabel("Output Video")
        self.output_label.setFont(bfont)
        self.output_path_show = QLabel("Path: ./outputs")
        self.output_path_show.setFixedHeight(50)
        self.output_path_show.setWordWrap(True)
        self.browse_output_button = QPushButton("Browse Output Folder")
        self.browse_output_button.clicked.connect(self.browse_output_folder)

        self.device_label = QLabel("Select Device")
        self.device_label.setFont(bfont)
        self.device_tree = QTreeView()
        self.device_model = QStandardItemModel()
        for device in self.device_list:
            device_name = torch.cuda.get_device_name(device) if device != "cpu" else "CPU"
            item = QStandardItem(f"{device_name} ({device})")
            self.device_model.appendRow(item)
        self.device_tree.setModel(self.device_model)
        self.device_tree.clicked.connect(self.device_selected)
        # 预先选中第一个
        self.device_tree.setCurrentIndex(self.device_model.index(0, 0))
        # 设置一个输入框用来设置linear gain

        self.linear_gain_label = QLabel("Linear Gain")
        self.linear_gain_label.setFont(bfont)
        self.linear_gain_input = QLineEdit("20")
        self.linear_gain_input.setValidator(QDoubleValidator())
        self.linear_gain_input.textEdited.connect(lambda x: setattr(self, "linear_gain", float(x) if x != "" else 0))
        self.linear_gain_set_button = QPushButton("Set Linear Gain")
        self.linear_gain_set_button.clicked.connect(self.set_linear_gain)

        # 设置一个输入框用来设置linear gain
        self.match_output_label = QLabel("Match Output Intensity")
        self.match_output_label.setFont(bfont)
        self.match_output_input = QCheckBox("Match Intensities of 2 Channels")
        self.match_output_input.setChecked(False)
        self.match_output_input.setFixedHeight(50)
        self.match_output_input.checkStateChanged.connect(lambda x: setattr(self, "match_output", x == Qt.CheckState.Checked))

        self.model_path_label = QLabel("Model File")
        self.model_path_label.setFont(bfont)
        self.model_path_show = QLabel("Path: ./GUI_Utils/Model/Fn15_Model.pkl")
        self.model_path_show.setWordWrap(True)
        self.model_path_show.setFixedHeight(50)
        self.model_path_show.setAcceptDrops(True)
        self.model_path_show.installEventFilter(self)
        self.browse_model_button = QPushButton("Browse MUFFLE model file")
        self.browse_model_button.clicked.connect(self.browse_model)

        self.choose_channel_label = QLabel("Choose Channel to Enhance")
        self.choose_channel_label.setFont(bfont)
        self.ch_a_box = QCheckBox("Acceptor")
        self.ch_b_box = QCheckBox("Donor")
        self.ch_a_box.setChecked(True)
        self.ch_b_box.setChecked(True)
        self.ch_a_box.checkStateChanged.connect(self.changeChnAChecked)
        self.ch_b_box.checkStateChanged.connect(self.changeChnBChecked)

        self.enhance_button = QPushButton("Reconstruct with MUFFLE")
        self.enhance_button.setFixedHeight(100)
        self.enhance_button.clicked.connect(self.enhance_videos)

        # 进度条
        self.enhance_progress = QProgressBar()
        self.enhance_progress.setRange(0, 100)

    def create_layout(self):
        central_widget = QWidget()
        layout = QHBoxLayout()
        in_layout = QVBoxLayout()
        out_layout = QVBoxLayout()
        setting_layout = QVBoxLayout()
        gain_layout = QHBoxLayout()
        gain_value_layout = QVBoxLayout()
        match_output_layout = QVBoxLayout()
        model_layout = QHBoxLayout()
        model_path_layout = QVBoxLayout()
        output_path_layout = QVBoxLayout()
        select_channel_layout = QHBoxLayout()

        in_layout.addWidget(self.input_label1)
        in_layout.addWidget(self.input_path1)
        in_layout.addWidget(self.input_video1)
        in_layout.addWidget(self.in1_sld)
        in_layout.addWidget(self.output_label1)
        in_layout.addWidget(self.output_video1)
        in_layout.addWidget(self.out1_sld)

        out_layout.addWidget(self.input_label2)
        out_layout.addWidget(self.input_path2)
        out_layout.addWidget(self.input_video2)
        out_layout.addWidget(self.in2_sld)
        out_layout.addWidget(self.output_label2)
        out_layout.addWidget(self.output_video2)
        out_layout.addWidget(self.out2_sld)
        layout.addLayout(in_layout)
        layout.addLayout(out_layout)

        setting_layout.addWidget(self.instructions_label)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))
        gain_value_layout.addWidget(self.linear_gain_label)
        gain_value_layout.addWidget(self.linear_gain_input)
        gain_value_layout.addWidget(self.linear_gain_set_button)
        match_output_layout.addWidget(self.match_output_label)
        match_output_layout.addWidget(self.match_output_input)
        gain_layout.addLayout(gain_value_layout)
        gain_layout.addLayout(match_output_layout)
        setting_layout.addLayout(gain_layout)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))

        model_path_layout.addWidget(self.model_path_label)
        model_path_layout.addWidget(self.model_path_show)
        model_path_layout.addWidget(self.browse_model_button)

        output_path_layout.addWidget(self.output_label)
        output_path_layout.addWidget(self.output_path_show)
        output_path_layout.addWidget(self.browse_output_button)

        model_layout.addLayout(model_path_layout)
        model_layout.addLayout(output_path_layout)
        setting_layout.addLayout(model_layout)

        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))

        setting_layout.addWidget(self.choose_channel_label)
        select_channel_layout.addWidget(self.ch_a_box)
        select_channel_layout.addWidget(self.ch_b_box)
        setting_layout.addLayout(select_channel_layout)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))
        setting_layout.addWidget(self.device_label)
        setting_layout.addWidget(self.device_tree)
        # 增加分割线
        setting_layout.addWidget(QLabel("----------------------------------------------------------------------------"))
        setting_layout.addWidget(self.enhance_progress)
        setting_layout.addWidget(self.enhance_button)
        layout.addLayout(setting_layout)

        central_widget.setLayout(layout)
        # 设置不允许窗口拉伸
        self.setFixedSize(1000, 720)
        self.setCentralWidget(central_widget)

    def pad_to_square(self, show_image):
        max_h_w = max(show_image.shape)
        padded_image = np.zeros((max_h_w, max_h_w), dtype=np.uint8)
        pad_h, pad_w = (max_h_w - show_image.shape[0]) // 2, (max_h_w - show_image.shape[1]) // 2
        padded_image[pad_h : pad_h + show_image.shape[0], pad_w : pad_w + show_image.shape[1]] = show_image
        return padded_image

    def changeChnAChecked(self, state):
        self.enhance_a = state == Qt.CheckState.Checked
        if not self.enhance_a and not self.enhance_b:
            self.enhance_b = True
            self.ch_b_box.setChecked(True)
        self.changeValueIn1(self.in1_sld.value())
        print(f"Enhance A: {self.enhance_a}, Enhance B: {self.enhance_b}")

    def changeChnBChecked(self, state):
        self.enhance_b = state == Qt.CheckState.Checked
        if not self.enhance_a and not self.enhance_b:
            self.enhance_a = True
            self.ch_a_box.setChecked(True)
        self.changeValueIn2(self.in2_sld.value())
        print(f"Enhance A: {self.enhance_a}, Enhance B: {self.enhance_b}")

    def changeValueIn1(self, value):
        # print(f"Slider value: {value}")
        if self.ch_a_numpy is not None and value >= 0:
            if value < self.ch_a_numpy.shape[0]:
                show_image = np.clip(self.ch_a_numpy[value] * self.linear_gain * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_a
                h, w = show_image.shape
                self.input_video1.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeValueIn2(self, value):
        # print(f"Slider value: {value}")
        if self.ch_b_numpy is not None and value >= 0:
            if value < self.ch_b_numpy.shape[0]:
                show_image = np.clip(self.ch_b_numpy[value] * self.linear_gain * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_b
                h, w = show_image.shape
                self.input_video2.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeValueOut1(self, value):
        # print(f"Slider value: {value}")
        if self.ch_a_out_numpy is not None and value >= 0:
            if value < self.ch_a_out_numpy.shape[0]:
                show_image = np.clip(self.ch_a_out_numpy[value] * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_a
                h, w = show_image.shape
                self.output_video1.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def changeValueOut2(self, value):
        # print(f"Slider value: {value}")
        if self.ch_b_out_numpy is not None and value >= 0:
            if value < self.ch_b_numpy.shape[0]:
                show_image = np.clip(self.ch_b_out_numpy[value] * 255, 0, 255).astype(np.uint8)
                show_image = self.pad_to_square(show_image) * self.enhance_b
                h, w = show_image.shape
                self.output_video2.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))

    def set_linear_gain(self):
        self.changeValueIn1(self.in1_sld.value())
        self.changeValueIn2(self.in2_sld.value())

    def loadAndSetInputPath(self, file_path, name="a"):
        if name == "a":
            in_slider = self.in1_sld
            out_slider = self.out1_sld
            path_label = self.input_path1
            video_viewer = self.input_video1
        else:
            in_slider = self.in2_sld
            out_slider = self.out2_sld
            path_label = self.input_path2
            video_viewer = self.input_video2

        path_label.setText("Path: " + file_path)
        data = load_tiff_seq(file_path)
        print(f"numpy.shape: {data.shape}")
        in_slider.setMaximum(data.shape[0] - 1)
        in_slider.setMinimum(0)
        out_slider.setMaximum(data.shape[0] - 1)
        out_slider.setMinimum(0)

        show_image = np.clip(data[0] * self.linear_gain * 255, 0, 255).astype(np.uint8)
        show_image = self.pad_to_square(show_image)
        h, w = show_image.shape
        video_viewer.setPixmap(QPixmap.fromImage(QImage(show_image, w, h, w, QImage.Format.Format_Grayscale8)))
        return data

    def eventFilter(self, obj, event):
        if hasattr(self, "input_video1") and hasattr(self, "input_video2"):
            if obj == self.input_video1 or obj == self.input_video2:
                if event.type() == QEvent.Type.DragEnter:
                    if event.mimeData().hasUrls():
                        event.acceptProposedAction()
                elif event.type() == QEvent.Type.Drop:
                    urls = event.mimeData().urls()
                    if urls:
                        file_path = urls[0].toLocalFile()
                        if obj == self.input_video1:
                            self.ch_a_numpy = self.loadAndSetInputPath(file_path, "a")
                        elif obj == self.input_video2:
                            self.ch_b_numpy = self.loadAndSetInputPath(file_path, "b")
        return super().eventFilter(obj, event)

    def browse_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder = folder_path
            self.output_path_show.setText("Path: " + folder_path)
            print(f"Output folder: {self.output_folder}")

    def browse_model(self):
        try:
            base_dir = QUrl.fromLocalFile("./GUI_Utils/Model")
            folder_path = QFileDialog.getOpenFileUrl(self, "Select Network Model", directory=base_dir, filter="*.pkl")
            path = folder_path[0].toLocalFile()
            yaml_path = path[: path.rfind(".")] + ".yaml"
            if os.path.exists(path) and os.path.exists(yaml_path):
                self.model_path = path
                self.model_path_show.setText("Path: " + path)
                self.opt_path = yaml_path
                print(f"Model Path: {self.output_folder}, opt_path: {self.opt_path}")
            else:
                # 弹出提示框
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Load model file error!")
                msg_box.setText(
                    f"Please make sure the model file (.pkl) and the corresponding configuration file (.yaml) are in the same folder. \nModel file: {path} \nYaml file: {yaml_path}"
                )
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.addButton(QMessageBox.StandardButton.Ok)
                msg_box.exec()
        except Exception as e:
            print(traceback.format_exc())

    def device_selected(self, index):
        print(f"Selected device: {self.device_model.data(index)}")
        self.selected_device = self.device_model.data(index)
        self.selected_device = self.selected_device[self.selected_device.find("(") + 1 : self.selected_device.find(")")]

    def enhance_videos(self):
        if self.selected_device == "cpu":
            # 弹出提示框
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Warning!")
            msg_box.setText(f"Running on CPU will be extremely slow. \nWe recommend using GPU to run MUFFLE. ")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.addButton(QMessageBox.StandardButton.Ok)
            msg_box.exec()

        # 调用另一个线程处理
        print(f"Linear Gain: {self.linear_gain}, Match Output: {self.match_output}, Device: {self.selected_device}, Enhance channel: A {self.enhance_a}, B {self.enhance_b}")
        t = ReconstructThread(
            self,
            self.opt_path,
            self.model_path,
            self.ch_a_numpy * self.linear_gain * self.enhance_a,
            self.ch_b_numpy * self.linear_gain * self.enhance_b,
            self.selected_device,
            self.enhance_progress,
            self.output_folder,
        )
        t.resultReady.connect(self.enhance_finished)
        t.updateProgress.connect(self.enhance_progress.setValue)
        t.error.connect(self.enhance_error)
        t.start()
        self.enhance_button.setEnabled(False)
        # t.quit()

    def enhance_finished(self, result):
        # print("Reconstruct finished: result", result)
        self.enhance_button.setEnabled(True)
        self.ch_a_out_numpy, self.ch_b_out_numpy, result_dir = result
        if self.match_output and self.enhance_a and self.enhance_b:
            self.ch_b_out_numpy = self.ch_b_out_numpy / np.mean(self.ch_b_out_numpy) * np.mean(self.ch_a_out_numpy)
        self.ch_a_out_numpy *= self.enhance_a
        self.ch_b_out_numpy *= self.enhance_b

        self.changeValueOut1(self.out1_sld.value())
        self.changeValueOut2(self.out2_sld.value())
        # 弹出提示框
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Success!")
        msg_box.setText(f"Video Reconstruction Finished! \nReconstructed videos are saved to {os.path.abspath(result_dir)}")
        msg_box.setIcon(QMessageBox.Icon.Information)
        open_folder_button = QPushButton("  Open folder  ")
        msg_box.addButton(open_folder_button, QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Ok)
        # 转换为绝对路径
        output_folder_abs = os.path.abspath(result_dir)
        open_folder_button.clicked.connect(lambda: subprocess.Popen(f'explorer /select, "{output_folder_abs}"'))
        msg_box.exec()
        self.enhance_progress.setValue(0)

    def enhance_error(self, result):
        # 弹出提示框
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Error!")
        msg_box.setText(f"Video Reconstruction Error! Please check the selected model and device. \n{str(result)}")
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.addButton(QMessageBox.StandardButton.Ok)
        msg_box.exec()
        self.enhance_progress.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoReconstructionApp()
    window.show()
    sys.exit(app.exec())
