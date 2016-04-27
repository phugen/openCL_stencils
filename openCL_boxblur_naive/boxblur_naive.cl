// An naive openCL implementation of a box blur filter.

// OpenCL kernel program. Takes an intensity image represented by width * height
// integer values. Then, it applies a box blur to the image and writes it
// to the output buffer.
__kernel void boxblur_naive (__global int* image,
                             __private int k,
                             __global int* output)
{
    // retrieve this work item's global work item id in x and y dimensions
    int col = get_global_id(0);
    int row = get_global_id(1);


    // check out-of-bounds conditions:
    // if the mask is out of bounds in at
    // least one direction, the work item yields.
    if(col - k < 0 ||
       col + k >= get_global_size(0) ||
       row - k < 0 ||
       row + k >= get_global_size(1))
    {
        return;
    }

    else
    {
        int sum = 0; // sum of all mask elements
        int maskElements = 0; // number of mask elements

        // get sum of all elements inside the mask
        // centered at the (col, row)
        for(int c_row = row - k; c_row <= row + k; c_row++)
            for(int c_col = col - k; c_col <= col + k; c_col++)
            {
                sum += image[c_col + c_row * get_global_size(0)]; // read pixel value at position (col, row)
                maskElements++;
            }

        // divide by size of mask
        int pixelValue = sum / maskElements;

        // write new pixel intensity value to output image
        output[col + row * get_global_size(0)] = pixelValue;
    }
}
