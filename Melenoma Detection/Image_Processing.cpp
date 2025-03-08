// zaib un nisa 21i-0383
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

const char* kernelSource =
"__kernel void grayscale(__global uchar4* inputImage, __global uchar* outputImage, uint width, uint height) { \n"
"    int x = get_global_id(0); \n"
"    int y = get_global_id(1); \n"
"    int index = y * width + x; \n"
"    uchar4 pixel = inputImage[index]; \n"
"    uchar grayscaleValue = (uchar)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z); \n"
"    outputImage[index] = grayscaleValue; \n"
"} \n";

unsigned char* loadImageData(const char* filename, size_t* width, size_t* height) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty()) {
        printf("Error opening image file: %s\n", filename);
        return NULL;
    }

    *height = image.rows;
    *width = image.cols;

    unsigned char* imageData = (unsigned char*)malloc(image.rows * image.cols * 4);
    if (!imageData) {
        printf("Error allocating memory for image data\n");
        return NULL;
    }

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            imageData[y * image.cols * 4 + x * 4 + 0] = pixel[0]; // Blue
            imageData[y * image.cols * 4 + x * 4 + 1] = pixel[1]; // Green
            imageData[y * image.cols * 4 + x * 4 + 2] = pixel[2]; // Red
            imageData[y * image.cols * 4 + x * 4 + 3] = 0; 
        }
    }

    return imageData;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input directory> <output directory>\n", argv[0]);
        return -1;
    }

    char* inputDir = argv[1];
    char* outputDir = argv[2];

    // Load OpenCL kernel code from string
    cl_int error;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    cl_command_queue_properties properties = 0;
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, &properties, &error);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &error);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "grayscale", &error);

    DIR* dir = opendir(inputDir);
    if (!dir) {
        perror("Error opening input directory");
        return -1;
    }

    mkdir(outputDir, 0777);

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        char inputPath[256];
        snprintf(inputPath, sizeof(inputPath), "%s/%s", inputDir, entry->d_name);

        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        if (strstr(entry->d_name, ".jpg") == NULL && strstr(entry->d_name, ".jpeg") == NULL)
            continue;

        size_t width, height;
        unsigned char* imageData = loadImageData(inputPath, &width, &height);
        if (!imageData)
            continue;

        size_t outputSize = width * height;
        unsigned char* outputImage = (unsigned char*)malloc(outputSize);
        if (!outputImage) {
            printf("Error allocating memory for output image data\n");
            free(imageData);
            continue;
        }

        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * sizeof(cl_uchar4), imageData, &error);

        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, outputSize, NULL, &error);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
        clSetKernelArg(kernel, 2, sizeof(cl_uint), &width);
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &height);

        size_t globalWorkSize[2] = { width, height };
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, outputSize, outputImage, 0, NULL, NULL);

        cv::Mat grayscaleImage(height, width, CV_8UC1, outputImage);

        char outputPath[512];
        snprintf(outputPath, sizeof(outputPath), "%s/%s_grayscale.jpg", outputDir, entry->d_name); // Change the output file extension if necessary
        cv::imwrite(outputPath, grayscaleImage);

        printf("Saved grayscale image: %s\n", outputPath);

        free(imageData);
        free(outputImage);
        clReleaseMemObject(inputBuffer);
        clReleaseMemObject(outputBuffer);
    }

    closedir(dir);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
