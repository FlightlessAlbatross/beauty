import numpy as np
from scipy.signal import convolve2d


def create_circular_disc(circle_radius: int, overall_size: int) -> np.ndarray:
    # Create a 0/1 array of size 2*overall_size + 1 with an approximate circle of 1s in the middle, with radius circle_radius
    
    size = 2 * overall_size + 1  # Compute the matrix size based on the overall size
    disc = np.zeros((size, size), dtype=int)  # Initialize a zero matrix

    center = overall_size  # The center of the disc
    for y in range(size):
        for x in range(size):
            # Calculate the Euclidean distance from the center
            if np.sqrt((x - center) ** 2 + (y - center) ** 2) <= circle_radius:
                disc[y, x] = 1

    return disc



def get_zone_mean(data:np.ndarray, na_value:float, kernel:np.ndarray) -> np.ndarray:
    #apply a filter with a given kernel to a 2d array and return the mean value for each element of the data array
    kernel = kernel.astype(float)
    kernel /= kernel.sum()

    data_filled = np.where(data == na_value, 0, data)
    # Perform the convolution
    convolved_sum = convolve2d(data_filled, kernel, mode='valid')
    return convolved_sum


def main(data:np.ndarray, na_value:float, circle_radius: int, overall_size: int):
    return get_zone_mean(data, na_value, create_circular_disc(circle_radius, overall_size))



if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python blur_circular_disc.py <data:np.ndarray> <na_value:float> <circle_radius: int> <overall_size: int>")
        sys.exit(1)
    
    data = sys.argv[1]
    na_value = sys.argv[2]
    circle_radius = sys.argv[3]
    overall_size = sys.argv[4]

    main(data, na_value, circle_radius, overall_size)
