// An openCL kernel implementation of a box blur filter.

// Takes an intensity image represented by width * height
// integer values. Then, it applies a box blur to the image and writes it
// to the output buffer.

// Uses blocking optimization to achieve better performance as opposed to
// a naive implementation where the number of work items is equal to
// the number of pixels in the image.

// Uses local GPU memory to compute values more efficiently within work groups.

__kernel void boxblur (__read_only __global int* image,
							 __read_only __global int* imageSize,
                             __read_only __global int* k,
							 __read_only __global int* blockSize,
							 __read_write __local int* localmem,
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

			// copy current value into local memory
			// (notice: "offset" needed because local memory size
			// is equal to work group size extended by mask size)
			localmem[(left + get_local_id(0)) + ((up + (get_local_id(1)) * (left + get_local_size(0) + right)))] = image[col + (row * imageSize[0])];

			// if current value is on a work group border,
			// load missing neighbor values into local memory (non-overlapping!)
			int val;
			int localpos;

			// top left value in work group
			if(get_global_id(0) == 0 &&
			   get_global_id(1) == 0)
			{
				output[col + (row * imageSize[0])] = 1; // DEBUG - why are all values in block filled with this value? same below


				/*for(int y = get_global_id(1) - up; y < get_global_id(1); y++)
					for(int x = get_global_id(0) - left; x < get_global_id(0); x++)
					{
						// use neutral element if position is OOB in input image
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];

						// calculate position in local memory
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));

						// copy neighbor values from global to local memory
						localmem[localpos] = val;
					}*/
			}

			// top right
			else if(get_local_id(0) == get_local_size(0) - 1 &&
			        get_local_id(1) == 0)
			{
				output[col + (row * imageSize[0])] = 2;
				/*for(int y = get_global_id(1) - up; y < get_global_id(1); y++)
					for(int x = get_global_id(0) + 1; x <= get_global_id(0) + right; x++)
					{
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));
						localmem[localpos] = val;
					}*/
			}

			// bottom left
			else if (get_local_id(0) == 0 &&
			         get_local_id(1) == get_local_size(1) - 1)
			{
				output[col + (row * imageSize[0])] = 3;
				/*for(int y = get_global_id(1) + 1; y <= get_global_id(1) + down; y++)
					for(int x = get_global_id(0) - left; x < get_global_id(0); x++)
					{
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));
						localmem[localpos] = val;
					}*/
			}

			// bottom right
			else if (get_local_id(0) == get_local_size(0) - 1 &&
			         get_local_id(1) == get_local_size(1) - 1)
			{
				output[col + (row * imageSize[0])] = 4;
				/*for(int y = get_global_id(1) + 1; y <= get_global_id(1) + down; y++)
					for(int x = get_global_id(0) + 1; x <= get_global_id(0) + right; x++)
					{
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));
						localmem[localpos] = val;
					}*/
			}

			// upper border
			else if (get_local_id(1) == 0)
			{
				output[col + (row * imageSize[0])] = 5;
				/*for(int y = get_global_id(1) - up; y < get_global_id(1); y++)
					for(int x = get_global_id(0) - left; x <= get_global_id(0) + right; x++) // ADD EXPLICIT CHECK HERE SO NO OVERLAP OCCURS
					{
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));
						localmem[localpos] = val;
					}*/
			}

			// right border
			else if (get_local_id(0) == get_local_size(0) - 1)
			{
				output[col + (row * imageSize[0])] = 6;
				/*for(int y = get_global_id(1) - up; y < get_global_id(1); y++)
					for(int x = get_global_id(0) + 1; x <= get_global_id(0) + right; x++) // ADD EXPLICIT CHECK HERE SO NO OVERLAP OCCURS
					{
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));
						localmem[localpos] = val;
					}*/
			}

			// lower border
			else if (get_local_id(1) == get_local_size(1) - 1)
			{
				output[col + (row * imageSize[0])] = 7;
				/*for(int y = get_global_id(1) + 1; y <= get_global_id(1) + down; y++)
					for(int x = get_global_id(0) - left; x <= get_global_id(0) + right; x++) // ADD EXPLICIT CHECK HERE SO NO OVERLAP OCCURS
					{
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));
						localmem[localpos] = val;
					}*/
			}

			// left border
			else if (get_local_id(0) == 0)
			{
				output[col + (row * imageSize[0])] = 8;
				/*for(int y = get_global_id(1) - up; y <= get_global_id(1) + down; y++)
					for(int x = get_global_id(0) - left; x < get_global_id(0); x++) // ADD EXPLICIT CHECK HERE SO NO OVERLAP OCCURS
					{
						val = y < 0 || x < 0 || y >= imageSize[1] || x >= imageSize[0] ? 0 : image[x + (y * imageSize[0])];
						localpos = (left + x) + ((up + y) * (left + get_local_size(0) + right));
						localmem[localpos] = val;
					}*/
			}


			// Normal case - value is not a border value.
			else
			{
				// nothing else to do
			}

			// only when all work items have arrived here,
			// computation continues - otherwise, not all needed
			// values might be available in local memory
			barrier (CLK_LOCAL_MEM_FENCE);


			// check out-of-bounds conditions
			// and resize mask if needed
			/*if (col - left < 0)
				left = left + (col - left);

			if (row - up < 0)
				up = up + (row - up);

			if (col + right >= imageSize[0])
				right = right - (col + right - imageSize[0]) - 1;

			if (row + down >= imageSize[1])
				down = down - (row + down - imageSize[1]) - 1;*/

			// get sum of all elements inside the mask
			// centered at (col, row)
			int sum = 0;
			int val2;

			for(int c_row = row; c_row <= up + row + down; c_row++)
				for(int c_col = col; c_col <= left + col + right; c_col++)
				{
					// check if value is out of bounds - if yes, use neutral element 0
					val2 = c_row < 0 ||
					       c_row >= up + get_local_size(1) + down ||
						   c_col < 0 ||
						   c_col >= left + get_local_size(0) + right ?
						   0 : localmem[c_col + c_row * (left + get_local_size(0) + right)];

					sum += val2; // sum neighbors using local memory
				}

			// divide by size of mask
			int masksize = (left + 1 + right) * (up + 1 + down); // +1 because of "middle" element
			int pixelValue = sum / masksize;

			// write new pixel intensity value to output image
			//output[col + (row * imageSize[0])] = localmem[(left + get_local_id(0)) + ((up + (get_local_id(1)) * (left + get_local_size(0) + right)))];
		}
	}
}
