// An naive openCL implementation of a box blur filter.

// OpenCL kernel program. Takes an intensity image represented by width * height
// integer values. Then, it applies a box blur to the image and writes it
// to the output buffer.




__kernel void boxblur_naive (__read_only __global int* image,
							 __read_only __global int* imageSize,//imageSize[0]: width of image, imageSize[1]: height
                             __read_only __global int* k,
							 __read_only __global int* blockSize,//blockSize[0]: width of block, blockSize[1]: height
                             __write_only __global int* output)
{
    int blockX = get_global_id(0) * blockSize[0];		//x-position of first element in block (block-intern coordinates (0,0))
	int blockY = get_global_id(1) * blockSize[1];		//y-position of first element in block

	//variables for block measures
	int blockWidth = blockSize[0];
	int blockHeight = blockSize[1];

	//calculation of values in block

	//check if block violates boundary on y-axis
	if (blockY + blockSize[1] >= imageSize[1])
		blockHeight = blockY + blockSize[1] - imageSize[1];	//reduce size of block to fit into y-boundary

	//calculate all positions in block
	for (int i = 0; i < blockHeight; i++)
	{
		//check if block violates boundary on x-axis
		if (blockX + blockSize[0] >= imageSize[0])
			blockWidth = blockX + blockSize[0] - imageSize[0];//resize block on x-axis to meet x-boundary

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

			// check out-of-bounds conditions:
			// if the mask is out of bounds in at
			// least one direction, the work item yields.
			if(col - left < 0 ||
				col + right >= imageSize[0] ||
				row - up < 0 ||
				row + down >= imageSize[1])
			{
				return;
			}

			else
			{
				//lines for testing purpose
				//int pixelValue = image[(blockY+i)*imageSize[0]+blockX+j]*2;
				//output[(blockY+i)*imageSize[0]+blockX+j] = pixelValue;
				int sum = 0; // sum of all mask elements
				int numMaskElements = 0; // number of mask elements

				// get sum of all elements inside the mask
				// centered at the (col, row)
				for(int c_row = row - up; c_row <= row + down; c_row++)
					for(int c_col = col - left; c_col <= col + right; c_col++)
					{
						sum += image[c_col + c_row * imageSize[0]]; // read pixel value at position (col, row)
						numMaskElements++;
					}

				// divide by size of mask
				int pixelValue = sum / numMaskElements;

				// write new pixel intensity value to output image
				//output[col + row * get_global_size(0)] = pixelValue;
				output[(blockX + j) + ((blockY + i) * imageSize[0])] = pixelValue;
			}
		}
	}
}
