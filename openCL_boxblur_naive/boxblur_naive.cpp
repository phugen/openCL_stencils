// the host program for boxblur_naive.cl.
// Depends on libjpeg for loading and saving JPEG images.

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
//#include "png_ops.hpp"

#define MASK_SIZE 1 // "k" parameter for box blur
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (10000)

#define IMAGE_WIDTH 10
#define IMAGE_HEIGHT 10

#define KERNEL_PATH "./boxblur_naive.cl"
#define KERNEL_NAME "boxblur_naive"

#define INPUT_FILENAME "alarm.jpg"
#define OUTPUT_FILENAME "alarm_blurred.jpg"

//using namespace cl;
using namespace std;

// read a file and convert it to a char*
static char* read_source (const char *filename)
{
    long int
        size = 0,
        res  = 0;

    char *src = NULL;

    FILE *file = fopen(filename, "rb");

    if (!file)  return NULL;

    if (fseek(file, 0, SEEK_END))
    {
        fclose(file);
        return NULL;
    }

    size = ftell(file);
    if (size == 0)
    {
        fclose(file);
        return NULL;
    }

    rewind(file);

    src = (char *)calloc(size + 1, sizeof(char));
    if (!src)
    {
        src = NULL;
        fclose(file);
        return src;
    }

    res = fread(src, 1, sizeof(char) * size, file);
    if (res != sizeof(char) * size)
    {
        fclose(file);
        free(src);

        return src;
    }

    src[size] = '\0'; /* NULL terminated */
    fclose(file);

    return src;
}

// create a matrix with random values
void createMatrix(uint8_t* matrix, uint8_t width, uint8_t height)
{
    srand (time(NULL)); // initialize random number seed

    // fill matrix with random values
    for(int i = 0; i < width * height; i++)
    {
        matrix[i] = rand() % 4;
    }

    // output original test matrix
    cout << "Original data:\n";
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            cout << +matrix[i * height + j] << " ";
        }

        cout << "\n";
    }
    cout << "\n\n\n";
}

int main (int argc, char* argv[])
{
    // open file containg kernel code
    char* source_str = read_source(KERNEL_PATH);

    // get length of source code in bytes
    size_t source_size = strlen(source_str) * sizeof(char);

    // Create OpenCL context with GPU as device
    cl_device_id device_id = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    uint8_t width = IMAGE_WIDTH;
    uint8_t height = IMAGE_HEIGHT;

    // get list of available platforms
    ret = clGetPlatformIDs(1, // max. number of platforms to find
                     &platform_id, // list of found openCL platforms
                     &ret_num_platforms); // number of found platforms

    if (ret != CL_SUCCESS)
        cout << "clGetPlatformIDs: " << ret << "\n";


    // get list of available devices
    ret = clGetDeviceIDs(platform_id, // the id of the platform to use
                   CL_DEVICE_TYPE_GPU, // type of device to use
                   1, // max number of devices to be used
                   &device_id, // list of found device IDs
                   &ret_num_devices); // number of found device IDs

    if (ret != CL_SUCCESS)
       cout << "clGetDeviceIDs: " << ret << "\n";


    // create openCL context
    cl_context context = clCreateContext(NULL, // list of context property names - NULL == implementation-defined
                                         1, // number of devices in list below
                                         &device_id, // list of devices to add to context
                                         NULL, // callback function pointer for error reporting function
                                         NULL, // callback function arguments
                                         &ret); // return value

    if (ret != CL_SUCCESS)
        cout << "clCreateContext: " << ret << "\n";


    // Create program "object"
    cl_program program_boxblur = clCreateProgramWithSource(context,
                                                            1, // number of program strings
                                                            (const char **) &source_str, // kernel source code
                                                            (const size_t *) &source_size, // size of string
                                                            &ret); // return value

    if (ret != CL_SUCCESS)
        cout << "clCreateProgramWithSource: " << ret << "\n";

    // Compile openCL kernel
    char build_params[] = {"-Werror"}; // treat warnings as errors

    ret = clBuildProgram (program_boxblur, // program object
                           1, // number of devices
                           &device_id, // list of devices
                           build_params, // compiler options
                           NULL, // callback function pointer for debug output
                           NULL); // callback function arguments


    // if build failed, output debug info
    if (ret != CL_SUCCESS)
    {
        cout << "clBuildProgram: " << ret << "\n";

        size_t len = 0;
        char* buffer;

        // check length of build log string
        clGetProgramBuildInfo(program_boxblur, // program object to request info on
                              device_id, // lol
                              CL_PROGRAM_BUILD_LOG, // type of information to request
                              0, // memory pointer to which information is written
                              NULL, // size of returned information
                              &len); // pointer to memory where to save length of returned string

        // allocate sufficient buffer - black cast magic due to compiling C with g++
        buffer = static_cast<char*>(calloc(len, sizeof(char)));

        // copy build log to buffer
        clGetProgramBuildInfo(program_boxblur, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

        // print build log
        fprintf(stderr, "%s\n", buffer);

        free(buffer);
    }



    // Select device and create a command queue for it
    cl_command_queue command_queue = clCreateCommandQueue(context,
                                                            device_id,
                                                            0, // properties -NULL is default
                                                            &ret); // return value
    if (ret != CL_SUCCESS)
        cout << "clCreateCommandQueue: " << ret << "\n";

    // load boxblur program into kernel
    cl_kernel kernel_boxblur = clCreateKernel(program_boxblur, // program variable
                                              KERNEL_NAME, // name of kernel main function
                                              &ret); // return value pointer

    if (ret != CL_SUCCESS)
        cout << "clCreateKernel: " << ret << "\n";

    // prepare kernel arguments
    uint8_t* h_testValues = (uint8_t*) malloc (width * height * sizeof(uint8_t)); // host memory for input image
    uint8_t* h_blurred = (uint8_t*) malloc (width * height * sizeof(uint8_t)); // host memory for output image

    // initialize allocated host memory with data
    createMatrix (h_testValues, IMAGE_WIDTH, IMAGE_HEIGHT); // create random 10x10 matrix
    memcpy((void *) h_blurred, (void *) h_testValues, width * height * sizeof(uint8_t)); // output matrix = input matrix


    // create openCL buffer objects

    // image format
    cl_image_format format; // image format
    format.image_channel_order = CL_INTENSITY; // brightness values only
    format.image_channel_data_type = CL_SIGNED_INT32; // values in [0, 4294967296]

    // image description
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE2D; // 2D image
    desc.image_width = (size_t) IMAGE_WIDTH;
    desc.image_height = (size_t) IMAGE_HEIGHT;
    desc.image_row_pitch = (size_t) 0;
    desc.image_slice_pitch = (size_t) 0;
    desc.num_mip_levels = (cl_uint) 0;
    desc.num_samples = (cl_uint) 0;
    desc.buffer = NULL; // maybe change this? <-------------------------------------

    // create input image object
    cl_mem d_image = clCreateImage (context,
                   CL_MEM_READ_ONLY, // flags
                   &format, // image format
                   &desc, // image description
                   0, // host pointer to raw image data
                   &ret);  // pointer to return value

    if (ret != CL_SUCCESS)
        cout << "clCreateImage_INPUT: " << ret << "\n";


    // create output image object
    cl_mem d_blurred = clCreateImage (context,
                   CL_MEM_WRITE_ONLY, // flags
                   &format, // image format
                   &desc, // image description
                   0, // host pointer to raw image data
                   &ret);  // pointer to return value

    if (ret != CL_SUCCESS)
        cout << "clCreateImage_OUTPUT: " << ret << "\n";


    // enqueue buffer transfers to kernel in command queue for buffers which are READ by the kernel
    const size_t origin[3] = {0, 0, 0}; // image origin offset
    const size_t region[3] = {IMAGE_WIDTH, IMAGE_HEIGHT, 1}; // width, height, depth

    // write input image to kernel
    ret = clEnqueueWriteImage(command_queue,
                               d_image, // image memory object
                               CL_TRUE, // blocking mode
                               origin, // origin (offset)
                               region, // dimensions
                               0, // row pitch - length of row in bytes, 0 default mode
                               0, // slice pitch - length ofslices in bytes, 0 default mode
                               h_testValues, // host pointer in memory
                               0, // number of events before this
                               NULL, // list of events to be executed before this
                               NULL); // event handle to this write action
    if (ret != CL_SUCCESS)
        cout << "clEnqueueWriteBuffer_INPUT: " << ret << "\n";

    // write output image to kernel
    ret = clEnqueueWriteImage(command_queue, d_blurred, CL_TRUE, origin, region, 0,  0,  h_blurred, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        cout << "clEnqueueWriteBuffer_OUTPUT: " << ret << "\n";


    // set kernel arguments
    ret = clSetKernelArg(kernel_boxblur, // kernel "object"
                   0, // argument index
                   sizeof(cl_mem), // size of argument (in this case: image buffer)
                   (void*) &d_image); // image to blur
    if (ret != CL_SUCCESS)
       cout << "clSetKernelArg_0: " << ret << "\n";

    // set mask size normally - without cl_mem object creation + write buffer
    uint8_t k = MASK_SIZE;
    ret = clSetKernelArg(kernel_boxblur, 1, sizeof(uint8_t), (void*) &k); // size of mask in pixels
    if (ret != CL_SUCCESS)
        cout << "clSetKernelArg_1: " << ret << "\n";

    ret = clSetKernelArg(kernel_boxblur, 2, sizeof(cl_mem), (void*) &d_blurred); // output image
    if (ret != CL_SUCCESS)
        cout << "clSetKernelArg_2: " << ret << "\n";

    // enqueue kernel and run it
    const size_t globalSizes[2] = {width, height};
    const size_t localSize[2] = {1, 1};

    ret = clEnqueueNDRangeKernel(command_queue, // command queue in which to enqueue task
                                    kernel_boxblur, // kernel to enqueue
                                    2, // number of work item dimensions
                                    NULL, // reserved for future use
                                    globalSizes, // number of work items per dimension
                                    localSize, // number of work items per work group per dimension
                                    0, // number of events that shall be executed before this one
                                    NULL, // list of events to be executed before this one - not needed
                                    NULL); // event handle - not needed


    if (ret != CL_SUCCESS)
        cout << "clEnqueueNDRangeKernel: " << ret << "\n";

    // wait for command queue to finish before reading results
    ret = clFinish(command_queue);

    // read from device and transfer image back to host
    ret = clEnqueueReadImage(command_queue,
                               d_blurred, // image memory object
                               CL_TRUE, // blocking mode
                               origin, // origin (offset)
                               region, // dimensions
                               0, // row pitch - length of row in bytes, 0 default mode
                               0, // slice pitch - length ofslices in bytes, 0 default mode
                               h_blurred, // host pointer in memory
                               0, // number of events before this
                               NULL, // list of events to be executed before this
                               NULL); // event handle to this write action

    // output blurred test matrix
    cout << "Changed data:\n";
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            cout << +h_blurred[i * height + j] << " ";
        }

        cout << "\n";
    }

    // release OpenCL resources
   clReleaseMemObject(d_image);
   clReleaseMemObject(d_blurred);

   clReleaseProgram(program_boxblur);
   clReleaseKernel(kernel_boxblur);
   clReleaseCommandQueue(command_queue);
   clReleaseContext(context);

   //release host memory
   free(h_testValues);
   free(h_blurred);


    return 0;
}
