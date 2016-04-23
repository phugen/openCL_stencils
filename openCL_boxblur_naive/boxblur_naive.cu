// An naive openCL implementation of a box blur filter.

// define an image sampler that deals with image boundaries automatically
// here: use value of border pixel that is closest to out-of-bounds pixel.
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// OpenCL kernel program. Takes a greyscale image, represented as a 2D array
// of integer values in a range from 0 - 255, a mask size k and writes its
// results into the output image array.
__kernel void boxblur (__read_only image2d_t image,
                       __private uint8_t k,
                       __global uint8_t* output)
{
    // retrieve this work item's global work item id in x and y dimensions
    const int2 pos = {get_global_id(0), get_global_id(1)};

    // calculate new pixel value from neighbor values
    // and respect image borders by not calculating border pixels
    uint32_t sum = 0;

    for(int i = pos.x - k; i < pos.x + k+1; i++)
    {
        for(int j = pos.y - k; j < pos.y + k+1; j++)
        {
            // add mask vector position (i, j) to position of current pixel
            // read_imagef returns a 4-vector where x is the intensity value.
            sum += *read_imagef(image, sampler, pos + (int2)(i, j)).x
        }
    }

    // clamp to uint8_t value space
    uint8_t pixelValue = clamp(sum, 0, 255);

    // divide by size of mask
    pixelValue =/ (k * k);

    // write new pixel value to output image
    output[pos[0]][pos[1]] = pixelValue;


    return;
}
