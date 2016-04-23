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


    // load PNG file and extract raw data
    void read_png_file(char* file_name)
    {
            char header[8];    // 8 is the maximum size that can be checked

            /* open file and test for it being a png */
            FILE *fp = fopen(file_name, "rb");
            if (!fp)
                    abort_("[read_png_file] File %s could not be opened for reading", file_name);
            fread(header, 1, 8, fp);
            if (png_sig_cmp(header, 0, 8))
                    abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);


            /* initialize stuff */
            png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

            if (!png_ptr)
                    abort_("[read_png_file] png_create_read_struct failed");

            info_ptr = png_create_info_struct(png_ptr);
            if (!info_ptr)
                    abort_("[read_png_file] png_create_info_struct failed");

            if (setjmp(png_jmpbuf(png_ptr)))
                    abort_("[read_png_file] Error during init_io");

            png_init_io(png_ptr, fp);
            png_set_sig_bytes(png_ptr, 8);

            png_read_info(png_ptr, info_ptr);

            width = png_get_image_width(png_ptr, info_ptr);
            height = png_get_image_height(png_ptr, info_ptr);
            color_type = png_get_color_type(png_ptr, info_ptr);
            bit_depth = png_get_bit_depth(png_ptr, info_ptr);

            number_of_passes = png_set_interlace_handling(png_ptr);
            png_read_update_info(png_ptr, info_ptr);


            /* read file */
            if (setjmp(png_jmpbuf(png_ptr)))
                    abort_("[read_png_file] Error during read_image");

            row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
            for (y=0; y<height; y++)
                    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

            png_read_image(png_ptr, row_pointers);

            fclose(fp);
    }


    void write_png_file(char* file_name)
    {
            /* create file */
            FILE *fp = fopen(file_name, "wb");
            if (!fp)
                    abort_("[write_png_file] File %s could not be opened for writing", file_name);


            /* initialize stuff */
            png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

            if (!png_ptr)
                    abort_("[write_png_file] png_create_write_struct failed");

            info_ptr = png_create_info_struct(png_ptr);
            if (!info_ptr)
                    abort_("[write_png_file] png_create_info_struct failed");

            if (setjmp(png_jmpbuf(png_ptr)))
                    abort_("[write_png_file] Error during init_io");

            png_init_io(png_ptr, fp);


            /* write header */
            if (setjmp(png_jmpbuf(png_ptr)))
                    abort_("[write_png_file] Error during writing header");

            png_set_IHDR(png_ptr, info_ptr, width, height,
                         bit_depth, color_type, PNG_INTERLACE_NONE,
                         PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

            png_write_info(png_ptr, info_ptr);


            /* write bytes */
            if (setjmp(png_jmpbuf(png_ptr)))
                    abort_("[write_png_file] Error during writing bytes");

            png_write_image(png_ptr, row_pointers);


            /* end write */
            if (setjmp(png_jmpbuf(png_ptr)))
                    abort_("[write_png_file] Error during end of write");

            png_write_end(png_ptr, NULL);

            /* cleanup heap allocation */
            for (y=0; y<height; y++)
                    free(row_pointers[y]);
            free(row_pointers);

            fclose(fp);
    }






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
