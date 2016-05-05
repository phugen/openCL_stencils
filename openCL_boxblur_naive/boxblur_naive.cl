// An naive openCL kernel implementation of a box blur filter.

// Takes an intensity image represented by width * height
// integer values. Then, it applies a box blur to the image and writes it
// to the output buffer.

__kernel void boxblur (__read_only __global int* image,
							 __read_only __global int* imageSize,
                             __read_only __global int* k,
							 __read_only __global int* blockSize,
							 __read_write __local int* localmem, // not needed but kept so host code can stay the same
                             __write_only __global int* output)
{
	// retrieve this work item's global work item id in x and y dimensions
    int col = get_global_id(0);
    int row = get_global_id(1);

    // extract mask dimensions for
    // easier use
    int left = k[0];
    int up = k[1];
    int right = k[2];
    int down = k[3];

	int sum = 0; // sum of all mask elements
	int val;

    // get sum of all elements inside the mask
    // centered at the (col, row)
    for(int c_row = row - up; c_row <= row + down; c_row++)
        for(int c_col = col - left; c_col <= col + right; c_col++)
        {
			// check if value is out of bounds - if yes, use neutral element 0
			val = c_row < 0 ||
				   c_row >= imageSize[1] ||
				   c_col < 0 ||
				   c_col >= imageSize[0] ?
				   0 : image[c_col + c_row * imageSize[0]];

			sum += val;
        }

    // divide by size of mask
	int masksize = (left + 1 + right) * (up + 1 + down); // +1 because of "middle" element
	int pixelValue = sum / masksize;

    // write new pixel intensity value to output image
    output[col + row * get_global_size(0)] = pixelValue;
}
