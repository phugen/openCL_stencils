// the host program for boxblur_naive.cl.
// Depends on libjpeg for loading and saving JPEG images.

#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "png_ops.hpp"

#define MASK_SIZE 5 // "k" parameter for box blur
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000) // approx. 1 MB

#define KERNEL_PATH "./boxblur_naive.cpp"
#define INPUT_FILENAME "alarm.jpg"
#define OUTPUT_FILENAME "alarm_blurred.jpg"

using namespace cl;

int main (int argc, char* argv[])
{
    // open file containg kernel code
    char string[MEM_SIZE];

    FILE *fp;
    char fileName[] = KERNEL_PATH;
    char *source_str;
    long unsigned int source_size;

    fp = fopen(fileName, "r");

    // check if file could be loaded
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    // convert to string
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);

    // close file
    fclose(fp);

    // Create OpenCL context with GPU as device
    cl_device_id device_id = NULL;
    cl_platform_id platform_id = NULL;

    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Compile OpenCL code
    cl_program program_boxblur = clCreateProgramWithSource(context, 1, (const char **) &source_str, (const long unsigned int *) &source_size, &ret);

    // Select device and create a command queue for it
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // load JPEG file and extract raw data
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo); // create libjpeg decompression object

    FILE* infile;
    if ((infile = fopen(INPUT_FILENAME, "rb")) == NULL) // load image file
    {
        fprintf(stderr, "can't open %s\n", FILENAME);
        exit(1);
    }

    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE); // read header information, i.e. dimensions and store them in jpeg object
    cinfo.out_color_space = JCS_EXT_RGBX; // fake alpha channel to create openCL-compatible 32 bit JPEG
    jpeg_start_decompress(&cinfo); // decompress JPEG

    // save image dimensions for openCL image
    // is this correct? image_width is of type "JDIMENSION"
    uint16_t width = cinfo.image_width;
    uint16_t height = cinfo.image_height;

    // image_components refers to the number of channels, i.e.
    // RGB vs. RGBA vs. pixel intensity...
    unsigned char* raw_data = unsigned char[cinfo.image_width * cinfo.image_components * cinfo.image_height];
    unsigned char* ptr = &raw_data;

    // (output_scanline keeps track of the number of scanlines
    // extracted so far)
    while(cinfo.output_scanline < cinfo.output_height)
    {
        jpeg_read_scanlines(&cinfo, &ptr, 1); // read a line
        ptr += cinfo.image_width * cinfo.image_components; // advance by one line
    }

    // finish decompression
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    // close file
    fclose(infile);


    // Create an OpenCL image and transfer data to the device
    const cl_image_format format; // image format
    format.image_channel_order = CL_INTENSITY; // brightness values only
    format.image_channel_data_type = CL_UNORM_INT8; // values in [0, 255]

    // create 2D image
    cl_int error = NULL;
    cl_mem image = clCreateImage2D (context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, // flags
                                    &format, // image format
                                    width, // image width in pixels
                                    height, // image height in pixels
                                    0, // number of bytes between start of two lines ("pitch")
                                    (void*) &raw_data, // host pointer to raw image data
                                    &error); // error value

    // Create a buffer for the result data
    Buffer blurred = Buffer(context,
                            CL_MEM_WRITE_ONLY, // access mode
                            sizeof(uint8_t) * width * height); // size of buffer

    // load boxblur program into kernel
    cl_kernel kernel_boxblur = clCreateKernel(program_boxblur, // program variable
                                              "boxblur_naive", // name of kernel file without extension
                                              &ret); // return value pointer

    // pass arguments to kernel
    kernel_boxblur.setArg(0, image); // image to blur
    kernel_boxblur.setArg(1, MASK_SIZE); // size of mask in pixels
    kernel_boxblur.setArg(2, blurred); // output image

    // enqueue kernel and thus run it
    ret = clEnqueueTask(command_queue, // command queue in which to enqueue task
                        kernel_boxblur, // kernel to enqueue
                        0, // number of events that shall be executed before this one
                        NULL, // list of events to be executed before this one - not needed
                        NULL); // event handle - not needed

    // read from device and transfer image back to host
    float* blurred_data = uint8_t[width * height]; // create buffer for reading data

    clEnqueueReadBuffer(command_queue, // command queue in which to enqueue task
                        blurred, // read buffer
                        CL_TRUE, // blocking read - function doesn't return until all data is read
                        0, // offset in bytes - start from beginning
                        sizeof(uint8_t) * width * height, // size of the data to be read
                        (void*) blurred_data, // pointer to buffer in host memory
                        0, // number of events that shall be executed before this one
                        NULL,  // list of events to be executed before this one - not needed
                        NULL);  // event handle - not needed



    // write output jpeg in grayscale
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    FILE* outfile;
    if ((outfile = fopen(OUTPUT_FILENAME, "wb")) == NULL) {
        fprintf(stderr, "can't open %s\n", OUTPUT_FILENAME);
        exit(1);
    }

    jpeg_stdio_dest(&cinfo, outfile);
    cinfo.image_width = width; // output width in pixels
    cinfo.image_height = height; // output height in pixels
    cinfo.input_components = 1;  // number color components per pixel - grayscale
    cinfo.in_color_space = JCS_GRAYSCALE; // color space - grayscale

    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE); // start compression

    JSAMPROW row_pointer[1]; // pointer to a single row
    int row_stride = image_width; // row width in buffer

    // write scanlines line by line, top to bottom
    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = &image_buffer[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // finish jpeg creation
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    // close file
    fclose(outfile);


    return 0;
}
