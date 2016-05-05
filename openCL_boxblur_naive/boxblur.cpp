// the host program for boxblur_x.cl.

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
//#include "png_ops.hpp"


// image size (power of 2)
#define IMAGE_WIDTH 8
#define IMAGE_HEIGHT 8

// "k" parameters for box blur
#define MASK_SIZE_LEFT 1
#define MASK_SIZE_UP 1
#define MASK_SIZE_RIGHT 1
#define MASK_SIZE_DOWN 1

// work item/group settings
#define LOCAL_X 4
#define LOCAL_Y 4
#define THREAD_NUM 4 // number of threads (defines block size)

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (10000)

// openCL paths
#define KERNEL_PATH "./boxblur_blocking_local.cl"
#define KERNEL_NAME "boxblur" // name of kernel function

#define INPUT_FILENAME "alarm.jpg"
#define OUTPUT_FILENAME "alarm_blurred.jpg"

using namespace std;



// display an error message if needed
void checkError (int ret, const char* funcName)
{
    if (ret != CL_SUCCESS)
        cout << funcName << ": " << ret << "\n";
}


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


// create a matrix with random values in the range [0, maxVal)
void createMatrix(cl_int* matrix, cl_int width, cl_int height, cl_int maxVal)
{
    srand (time(NULL)); // initialize random number seed

    // fill matrix with random values
    for(int i = 0; i < width * height; i++)
    {
        matrix[i] = rand() % maxVal;
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

    cl_int width = IMAGE_WIDTH;
    cl_int height = IMAGE_HEIGHT;

    // get list of available platforms
    ret = clGetPlatformIDs(1, // max. number of platforms to find
                     &platform_id, // list of found openCL platforms
                     &ret_num_platforms); // number of found platforms
    checkError(ret, "clGetPlatFormIDs");

    // get list of available devices
    ret = clGetDeviceIDs(platform_id, // the id of the platform to use
                   CL_DEVICE_TYPE_GPU, // type of device to use - can be changed to CPU for debugging
                   1, // max number of devices to be used
                   &device_id, // list of found device IDs
                   &ret_num_devices); // number of found device IDs
    checkError(ret, "clGetDeviceIDs");


    // create openCL context
    cl_context context = clCreateContext(NULL, // list of context property names - NULL == implementation-defined
                                         1, // number of devices in list below
                                         &device_id, // list of devices to add to context
                                         NULL, // callback function pointer for error reporting function
                                         NULL, // callback function arguments
                                         &ret); // return value
    checkError(ret, "clCreateContext");


    // Create program "object"
    cl_program program_boxblur = clCreateProgramWithSource(context,
                                                            1, // number of program strings
                                                            (const char **) &source_str, // kernel source code
                                                            (const size_t *) &source_size, // size of string
                                                            &ret); // return value
    checkError(ret, "clCreateProgramWithSource");

    // Compile openCL kernel
    char build_params[] = {"-Werror"}; // treat warnings as errors

    ret = clBuildProgram (program_boxblur, // program object
                           1, // number of devices
                           &device_id, // list of devices
                           build_params, // compiler options
                           NULL, // callback function pointer for debug output
                           NULL); // callback function arguments
    checkError(ret, "clBuildProgram");


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
    checkError(ret, "clCreateCommandQueue");

    // load boxblur program into kernel
    cl_kernel kernel_boxblur = clCreateKernel(program_boxblur, // program variable
                                              KERNEL_NAME, // name of kernel main function
                                              &ret); // return value pointer
    checkError(ret, "clCreateKernel");

    // prepare kernel argument host memory
    cl_int* h_testValues = (cl_int*) malloc (width * height * sizeof(cl_int)); // host memory for input image
	cl_int* h_matrixSize = (cl_int*) malloc (2 * sizeof(cl_int));	// host memory for matrix dimensions
    cl_int* h_masksize = (cl_int*) malloc (4 * sizeof(cl_int)); // host memory for mask dimensions
	cl_int* h_blocksize = (cl_int*) malloc (2 * sizeof(cl_int)); // host memory for block size
    cl_int* h_blurred = (cl_int*) malloc (width * height * sizeof(cl_int)); // host memory for output image

    // initialize allocated host memory with data
    createMatrix (h_testValues, IMAGE_WIDTH, IMAGE_HEIGHT, 4); // create random matrix
    memset((void*) h_blurred, 0, width * height * sizeof(cl_int)); // initialize output matrix as 0-matrix

	// set matrix dimensions
	h_matrixSize[0] = IMAGE_WIDTH;
	h_matrixSize[1] = IMAGE_HEIGHT;

    // set mask dimensions
    h_masksize[0] = MASK_SIZE_LEFT;
    h_masksize[1] = MASK_SIZE_UP;
    h_masksize[2] = MASK_SIZE_RIGHT;
    h_masksize[3] = MASK_SIZE_DOWN;

    // set block sizes based on number
    // of available threads
	h_blocksize[0] = (IMAGE_WIDTH) / THREAD_NUM;
	h_blocksize[1] = (IMAGE_HEIGHT) / THREAD_NUM;

    // create openCL buffer objects
    // create input buffer
    cl_mem d_image = clCreateBuffer (context,
  	                                 CL_MEM_READ_ONLY, // flags - read-only in kernel
  	                                 width * height * sizeof(cl_int), // size of buffer
  	                                 NULL, // host pointer to memory for buffer
  	                                 &ret); // return value
    checkError(ret, "clCreateBuffer_INPUT");

	cl_mem d_matrixSize = clCreateBuffer (context,
										  CL_MEM_READ_ONLY,
										  2 * sizeof(cl_int),
										  NULL,
										  &ret);
	checkError(ret, "clCreateBuffer_MATRIXSIZE");

    cl_mem d_masksize = clCreateBuffer (context,
                                        CL_MEM_READ_ONLY,
                                        4 * sizeof(cl_int),
                                        NULL,
                                        &ret);
    checkError(ret, "clCreateBuffer_MASKSIZE");

	cl_mem d_blocksize = clCreateBuffer (context,
										  CL_MEM_READ_ONLY,
										  2 * sizeof(cl_int),
										  NULL,
										  &ret);
	checkError(ret, "clCreateBuffer_BLOCKSIZE");

    // create output image object
    cl_mem d_blurred = clCreateBuffer (context,
                                       CL_MEM_WRITE_ONLY, // write-only in kernel
                                       width * height * sizeof(cl_int),
                                       NULL,
                                       &ret);
    checkError(ret, "clCreateBuffer_OUTPUT");


    // write input image to kernel
    ret = clEnqueueWriteBuffer(command_queue,
                               d_image, // buffer object
                               CL_TRUE, // blocking write
                               0, // offset
                               width * height * sizeof(cl_int), // size of data being written
                               (void*) h_testValues, // host pointer to data to be written
                               0, // number of events before this
                               NULL, // list of events to be executed before this
                               NULL); // event handle to this write action
    checkError(ret, "clEnqueueWriteBuffer_INPUT");

	// write matrixsize array to kernel
    ret = clEnqueueWriteBuffer(command_queue,
                               d_matrixSize,
                               CL_TRUE,
                               0,
                               2 * sizeof(cl_int),
                               (void*) h_matrixSize,
                               0,
                               NULL,
                               NULL);
    checkError(ret, "clEnqueueWriteBuffer_MATRIXSIZE");

    // write masksize array to kernel
    ret = clEnqueueWriteBuffer(command_queue,
                               d_masksize,
                               CL_TRUE,
                               0,
                               4 * sizeof(cl_int),
                               (void*) h_masksize,
                               0,
                               NULL,
                               NULL);
    checkError(ret, "clEnqueueWriteBuffer_MASKSIZE");

	// write blocksize array to kernel
    ret = clEnqueueWriteBuffer(command_queue,
                               d_blocksize,
                               CL_TRUE,
                               0,
                               2 * sizeof(cl_int),
                               (void*) h_blocksize,
                               0,
                               NULL,
                               NULL);
    checkError(ret, "clEnqueueWriteBuffer_BLOCKSIZE");

    // write output image to kernel
    ret = clEnqueueWriteBuffer(command_queue,
                               d_blurred,
                               CL_TRUE,
                               0,
                               width * height * sizeof(cl_int),
                               (void*) h_blurred,
                               0,
                               NULL,
                               NULL);
    checkError(ret, "clEnqueueWriteBuffer_OUTPUT");


    // set kernel arguments
    ret = clSetKernelArg(kernel_boxblur, // kernel "object"
                         0, // argument index
                         sizeof(cl_mem), // size of argument (in this case: image buffer)
                         (void*) &d_image); // image to blur
    checkError(ret, "clSetKernelArg_0");

	// set matrix size
    ret = clSetKernelArg(kernel_boxblur, 1, sizeof(cl_mem), (void*) &d_matrixSize); // size of matrix in XY dimensions
    checkError(ret, "clSetKernelArg_1");

    // set mask size normally - without cl_mem object creation + write buffer
    ret = clSetKernelArg(kernel_boxblur, 2, sizeof(cl_mem), (void*) &d_masksize); // size of mask in NSWE dimensions
    checkError(ret, "clSetKernelArg_2");

	// set block size
    ret = clSetKernelArg(kernel_boxblur, 3, sizeof(cl_mem), (void*) &d_blocksize); // size of block in XY dimensions
    checkError(ret, "clSetKernelArg_3");

    // implicitly allocate local memory by passing "NULL" instead of cl_mem object
    ret = clSetKernelArg(kernel_boxblur, 4, (size_t) (MASK_SIZE_LEFT + LOCAL_X + MASK_SIZE_RIGHT) * (MASK_SIZE_UP + LOCAL_Y + MASK_SIZE_DOWN), NULL);
    checkError(ret, "clSetKernelArg_4");

    ret = clSetKernelArg(kernel_boxblur, 5, sizeof(cl_mem), (void*) &d_blurred); // output image
    checkError(ret, "clSetKernelArg_5");

    // enqueue kernel and run it
	// set number of work items
	int globalX = IMAGE_WIDTH / h_blocksize[0];
	int globalY = IMAGE_HEIGHT / h_blocksize[1];
    const size_t globalSizes[2] = {globalX, globalY}; // number of work items per dimension
    const size_t localSize[2] = {LOCAL_X, LOCAL_Y}; // number of work items per work group per dimension

    cout << "Image size is X:" << IMAGE_WIDTH << " Y:" << IMAGE_HEIGHT << "\n";
    cout << "Number of threads is " << THREAD_NUM << "\n";
    cout << "Block sizes are X:" << h_blocksize[0] << " Y:" << h_blocksize[1] << "\n";
    cout << "Number of work items per dimension: X:" << globalX << " Y:" << globalY << "\n";
    cout << "Number of work items per work group per dimension: X:" << LOCAL_X << " Y:" << LOCAL_Y << "\n\n";

    ret = clEnqueueNDRangeKernel(command_queue, // command queue in which to enqueue task
                                    kernel_boxblur, // kernel to enqueue
                                    2, // number of work item dimensions
                                    NULL, // reserved for future use
                                    globalSizes, // number of work items per dimension
                                    localSize, // number of work items per work group per dimension
                                    0, // number of events that shall be executed before this one
                                    NULL, // list of events to be executed before this one - not needed
                                    NULL); // event handle - not needed
    checkError(ret, "clEnqueueNDRangeKernel");

    // wait for command queue to finish before reading results
    ret = clFinish(command_queue);
    checkError(ret, "clFinish");

    // read from device and transfer buffer back to host
    ret = clEnqueueReadBuffer (command_queue,
                               d_blurred, // buffer object
                               CL_TRUE, // blocking read
                               0, // offset
                               width * height * sizeof(cl_int), // size of buffer
                               (void *) h_blurred, // host pointer to memory where to write buffer contents
                               0, // number of events to complete before this
                               NULL, // list of events to complete
                               NULL); // event handle to this write action
    checkError(ret, "clEnqueueReadBuffer");

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
   clReleaseMemObject(d_matrixSize);
   clReleaseMemObject(d_masksize);
   clReleaseMemObject(d_blocksize);
   clReleaseMemObject(d_blurred);

   clReleaseProgram(program_boxblur);
   clReleaseKernel(kernel_boxblur);
   clReleaseCommandQueue(command_queue);
   clReleaseContext(context);

   // release host memory
   free(h_testValues);
   free(h_matrixSize);
   free(h_masksize);
   free(h_blocksize);
   free(h_blurred);


    return 0;
}
