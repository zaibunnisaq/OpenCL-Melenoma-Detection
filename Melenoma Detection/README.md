
This project focuses on utilizing **OpenCL for parallel computing** to convert colored images of skin lesions from the **ISIC 2020 Challenge Dataset** into grayscale images. The primary objective is to enhance computational efficiency while maintaining accuracy in medical image analysis.


## **Dataset**
- The dataset consists of **10,982 JPEG images** of skin lesions.
- It includes **benign and malignant lesions** from over **2,000 patients**.
- **Dataset Source:** [Download the ISIC 2020 Test Dataset](https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip)

---

## **Implementation Overview**
### **1. Environment Setup**
- **Install OpenCL SDK** and necessary GPU drivers.
- Setup OpenCL-supported environment using [OpenCL-and-Docker Repository](https://github.com/umarwaseeem/OpenCL-and-Docker) as a base.

### **2. OpenCL Implementation**
- **Parallel Processing:** Using OpenCL **kernels** to convert **RGB to grayscale** efficiently.
- **Handles Variable Image Sizes:** The implementation ensures dynamic memory allocation to process different image resolutions.
- **Optimized Memory Use:** Efficient use of **host, global, local, and private memory** to ensure fast execution.

### **3. Image Processing Pipeline**
- **Input:** Colored images from the ISIC dataset.
- **Processing:**
  - Load image into memory.
  - Apply OpenCL **grayscale conversion kernel**.
  - Store the processed grayscale image.
- **Output:** Grayscale images stored in `output_images/` folder.

---

## **Example Input and Output**
- **Input Image:** ![Sample Input]()
- **Output Image:** ![Sample Output]()

