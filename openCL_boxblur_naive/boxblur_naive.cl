// An openCL kernel implementation of a box blur filter.

// Takes an intensity image represented by width * height
// integer values. Then, it applies a box blur to the image and writes it
// to the output buffer.

// Uses blocking optimization to achieve better performance as opposed to
// a naive implementation where the number of work items is equal to
// the number of pixels in the image.

__kernel void boxblur_naive (__read_only __global int* image,
							 __read_only __global int* imageSize,
                             __read_only __global int* k,
							 __read_only __global int* blockSize,
                             __write_only __global int* output)
{
    int blockX = get_global_id(0) * blockSize[0]; // x position of first element in block (internal block coordinates (0,0))
	int blockY = get_global_id(1) * blockSize[1]; // y position of first element in block

	// get block sizes
	int blockWidth = blockSize[0];
	int blockHeight = blockSize[1];

	// check if block violates boundary on y-axis
	if (blockY + blockSize[1] > imageSize[1])
		blockHeight = blockY + (blockSize[1] - 1) - imageSize[1];	// reduce size of block to fit y-boundary

	// calculate all positions in block
	for (int i = 0; i < blockHeight; i++)
	{
		// check if block violates boundary on x-axis
		if (blockX + blockSize[0] > imageSize[0])
			blockWidth = blockX + (blockSize[0] - 1) - imageSize[0]; // reduce size of block to fit x-boundary

		for (int j = 0; j < blockWidth; j++)
		{
			// find position of matrix entry to calculate
			int col = blockX + j;
			int row = blockY + i;

			// extract mask dimensions for easier use
			int left = k[0];
			int up = k[1];
			int right = k[2];
			int down = k[3];

			// check out-of-bounds conditions
			// and resize mask if needed
			if (col - left < 0)
				left = left + (col - left);

			if (row - up < 0)
				up = up + (row - up);

			if (col + right >= imageSize[0])
				right = right - (col + right - imageSize[0]) - 1;

			if (row + down >= imageSize[1])
				down = down - (row + down - imageSize[1]) - 1;

			// get sum of all elements inside the mask
			// centered at (col, row)
			int sum = 0;

			for(int c_row = row - up; c_row <= row + down; c_row++)
				for(int c_col = col - left; c_col <= col + right; c_col++)
				{
					sum += image[c_col + c_row * imageSize[0]]; // read pixel value at position (col, row)
				}

			// divide by size of mask
			int masksize = (left + right + 1) * (up + down + 1); // +1 because of "middle" element
			int pixelValue = sum / masksize;

			// write new pixel intensity value to output image
			output[col + (row * imageSize[0])] = pixelValue;
		}
	}
}
