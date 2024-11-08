
# MUFFLE GUI Manual

This is a GUI application for our paper **"Supervised multi-frame dual-channel denoising enables long-term single-molecule FRET under extremely low photon budget"**. The application is designed to enhance the quality of low-light single-molecule FRET videos by using a pretrained MUFFLE model. The python scripts supports multiple platform and the compiled binary application runs on Windows 10/11 (Linux support in progress).

### TL;DR: Just click the "Reconstruct with MUFFLE" button to enhance the preloaded videos. 

![Alt text](GUI_Manual_Image\Image1.png)

## Compatibility & Installation

If you have a GPU which supports CUDA >= 12.1, we recommend **running binary application on GPU**. Otherwise please **run binary application on CPU**. If you have a GPU with CUDA < 12.1 or you want custom environment settings, please **run the python script** instead (you may still be able to run the binary but is not guaranteed).

1. **Running binary application on CPU**
   - No additional requirements needed.
   - Double-click the "MUFFLE_GUI.exe" file to launch the GUI.
   - **Note: The reconstruction process will be extremely slow on CPU. Not recommended for formal use.**
2. **Running binary application on GPU with CUDA >= 12.1 support (recommended)**
   - Make sure that the GPU device has CUDA support. Check [here](https://developer.nvidia.com/cuda-gpus) for a list of supported GPUs.
   - Install CUDA >= 12.1 from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64).
   - Double-click the "MUFFLE_GUI.exe" file to launch the GUI.
3. **Running the python script**
   - Install the following packages:
     - [PyQt5](https://pypi.org/project/PyQt5/)
     - [PyTorch](https://pytorch.org/)
     - [Numpy](https://numpy.org/)
     - [Tifffile](https://pypi.org/project/tifffile/)
     - Other packages: see [requirements.txt](requirements.txt)
   - Run the python script "MUFFLE_GUI.py" to launch the GUI.


We have tested the compatibility of the GUI application on the following platforms:
- Intel / AMD CPUs on Windows 10 / Windows 11
- NVIDIA GTX 1080Ti (driver 536.99) on Windows 11
- NVIDIA GTX 1050 (driver 546.33) on Windows 10
- Macbook with M1 CPU.
- **Report the compatibility** of the application on other platforms by adding an issue.


## Computation Time
We have tested the computation time of the GUI application on the following platforms:

| Type 	| Hardware          	| Computation Time ($160\times 180$ px) 	|
|------	|-------------------	|--------------------------------------	|
| CPU  	| Intel Core 10900X 	| **5.62 sec / frame**                       	|
| GPU  	| NVIDIA RTX 2080   	| **0.28 sec / frame**                       	|
| GPU  	| NVIDIA GTX 1080Ti 	| **0.24 sec / frame**                       	|


---

## Usage

MUFFLE GUI is a standalone application that can be run on Windows 10. The application can be run on CPU or GPU. 

### Step 1: Load Input Videos

- Drag and drop the Acceptor dark video file onto the "Acceptor dark" video viewer. The input video can **either be a single 3-Dim tiff file or a folder containing multiple tiff frames.**
- Drag and drop the Donor dark video file onto the "Donor dark" video viewer.
- The file paths for both videos will be displayed above their respective viewers.

![Alt text](GUI_Manual_Image\Image2.png)

---

### Step 2: Set Linear Gain


The linear gain is used to adjust the brightness of the input videos and is applied to both videos. **Please adjust the linear gain to ensure that the input videos are not too dark or too bright.** The following reconstruction process will maintain the brightness of the input videos. The default value is 10.

- Adjust the "Linear Gain" input field to control the linear gain for both videos.
- Use the "Set Linear Gain" button to apply the chosen gain.

![Alt text](GUI_Manual_Image\Image3.png)

---



### Step 3: View Videos
- Each frame of the input videos will be padded with a black border to a square shape.
- **Adjust the sliders** to navigate through the frames.
- If you find that the videos are too dark or too bright, please adjust the linear gain.
- After the reconstruction process, the reconstructed videos will be displayed in the "Reconstructed Acceptor" and "Reconstructed Donor" video viewers.

![Alt text](GUI_Manual_Image\Image4.png)

---


### Step 4: Set Model Path

We provide a pretrained MUFFLE model for dual-channel video enhancement. The model file is placed alongside the YAML configuration file in the "MUFFLE_Model" folder. 

We currently provide two models: 
- **Fn15_Model**: Trained to enhance both the Acceptor and Donor channels by considering the cross-talk between the two channels. Must input both Acceptor and Donor videos.
- **Fn15_NoFus_Model**: Trained without the cross-talk between the two channels. The reconstruction process is independent for each channel. The reconstruction quality is slightly lower. But can input only one channel or both channels.

To use a custom MUFFLE model, please refer to the [MUFFLE]() repository for details on training a custom model. Please ensure that the model file and its corresponding YAML configuration file are placed in the same folder.

**Set MUFFLE Model File:**
- Click the "Browse MUFFLE model file" button.
- Choose the MUFFLE model file (with a .pkl extension) and its corresponding configuration file (with a .yaml extension).
- Confirm the model file path displayed below.

![Alt text](GUI_Manual_Image\Image5.png)

---

### Step 5: Set Output Folder

The enhanced videos will be saved in the output folder as 3-Dim tiff files. The default output folder is the "Output" folder in the current directory.

**Set Output Folder:**
- Click the "Browse Output Folder" button.
- Select the folder where enhanced videos will be saved.
- Confirm the output folder path displayed below.

![Alt text](GUI_Manual_Image\Image6.png)


### Step 6: Choose Channel to Enhance

MUFFLE supports enhancing both the Acceptor and Donor channels. You can choose to enhance only one channel or both channels. However, make sure to load the correct model file, otherwise you make get bad reconstruction result. The supported model files for each of the following cases are:

|                  	| Acceptor and Donor 	| Only Acceptor 	| Only Donor 	|
|------------------	|--------------------	|---------------	|------------	|
| Fn15_Model       	| √                  	| ×             	| ×          	|
| Fn15_NoFus_Model 	| √                  	| √             	| √          	|

![Alt text](GUI_Manual_Image\Image7.png)
![Alt text](GUI_Manual_Image\Image8.png)

---

### Step 7: Select Device

We recommend using a GPU device for video enhancement. Running MUFFLE on CPU is supported but will be extremely slow. If you cannot find your GPU device in the list, please check that CUDA is installed correctly and your GPU device is supported (see Step 1).

- Choose a CPU/GPU device from the list of available devices.
- This device will be used for video enhancement.
![Alt text](GUI_Manual_Image\Image9.png)


---

### Step 8: Reconstruct with MUFFLE

- Click the **"Reconstruct with MUFFLE"** button.
- The progress bar will show the enhancement progress.
- Note: if you are running MUFFLE on CPU, the reconstruction process will be extremely slow. Not recommended for formal use.

![Alt text](GUI_Manual_Image\Image1.png)

---

### Step 9: Explore Output Folder

- Click the **"Open folder"** button in the success message to open the output folder in File Explorer.
- The reconstructed videos will be saved in this folder.
---

### Note:

- **Error Handling:**
  - If an error occurs during the reconstruction process, an error message will be displayed. Please review the error message for details.
  - If you encounter an error, please check that the input videos are in the correct format and the MUFFLE model file is valid.
  - If you are still unable to resolve the error, please contact us by adding an issue.

## Updates:

### Ver. 1.0 (2024-01-28)
- Initial release.
### Ver. 1.1 (2024-02-05)
- Catch exception: failure in preloading images.
### Ver. 1.2 (2024-02-07)
- Fix bug: failure in loading images of shape (P, H, W, 1).
### Ver. 1.3 (2024-05-31)
- Add support for single channel video enhancement.
  - Can choose to enhance only Acceptor or Donor channel.
  - Must choose at least one channel.
- Add support for matching output intensities of both channels.
- Change the default path for model file selection to GUI_Utils\Model.
- Adjust layout of the GUI.