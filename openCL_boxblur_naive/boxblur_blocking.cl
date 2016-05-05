// An openCL kernel implementation of a box blur filter.

// Takes an intensity image represented by width * height
// integer values. Then, it applies a box blur to the image and writes it
// to the output buffer.

// Uses blocking optimization to achieve better performance as opposed to
// a naive implementation where the number of work items is equal to
// the number of pixels in the image.

__kernel void boxblur (__read_only __global int* image,
							 __read_only __global int* imageSize,
                             __read_only __global int* k,
							 __read_only __global int* blockSize,
							 __read_write __local int* localmem, // not needed but kept so host code can stay unchanged
                             __write_only __global int* output)
{
	// extract mask dimensions for easier use
	int left = k[0];
	int up = k[1];
	int right = k[2];
	int down = k[3];

    int blockX = get_global_id(0) * blockSize[0]; // x position of first element in block (internal block coordinates (0,0))
	int blockY = get_global_id(1) * blockSize[1]; // y position of first element in block

	// get block sizes
	int blockWidth = blockSize[0];
	int blockHeight = blockSize[1];

	// calculate all positions in block
	for (int i = 0; i < blockHeight; i++)
	{
		for (int j = 0; j < blockWidth; j++)
		{
			// find position of global matrix entry to calculate
			int col = blockX + j;
			int row = blockY + i;

			// get sum of all elements inside the mask
			// centered at (col, row)
			int sum = 0;
			int val;

			for(int c_row = row - up; c_row <= row + down; c_row++)
				for(int c_col = col - left; c_col <= col + right; c_col++)
				{
					// check if value is out of bounds - if yes, use neutral element 0
					val = c_row < 0 ||
					       c_row >= imageSize[1] ||
						   c_col < 0 ||
						   c_col >= imageSize[0] ?
						   0 : image[c_col + (c_row * imageSize[0])];

					sum += val; // sum neighbors
				}

			// divide by size of mask
			int masksize = (left + 1 + right) * (up + 1 + down); // +1 because of "middle" element
			int pixelValue = sum / masksize;

			// write new pixel intensity value to output image
			output[col + (row * imageSize[0])] = pixelValue;
		}
	}
}
