import numpy as np



def crossCorrelation3D(input: np.ndarray, kernel: np.ndarray, stride: int, mode:str = "valid"):

    inpt = input
    inp_depth, inp_width, inp_height = input.shape
    kernel_depth, kernel_width, kernel_height  = kernel.shape
    #print(f"Shape: {input.shape}")

    w_pad = 0
    h_pad = 0

    if mode == "full":

        #print(f"full kernels {kernel.shape}")
        #print(f"full input{input.shape}")

        w_pad = kernel_width - 1
        h_pad = kernel_height - 1

        padded_input = []

        for i in range(inp_depth):

            padded_input.append(np.pad(inpt[i], (w_pad,h_pad)))

        inpt = np.array(padded_input)
        stride = 1

    #print("huh")
    out_height = ((inp_height - kernel_height + (2 * h_pad)) // stride) + 1
    out_width = ((inp_width - kernel_width + (2 * w_pad)) // stride) + 1


    output = np.zeros((out_width,out_height))

    #print(f"Conv input shape {input.shape}")
    #print(f"Conv output Shape {output.shape}")
    #print(f"Conv kernel Shape {kernel.shape}")
    
    for i in range(out_height):

        for j in range(out_width):

            for k in range(kernel_depth):                

                    #print(f"conv:  {i}, {j}, {d}")
                
                    output[i,j] += np.sum(inpt[:, i*stride : i*stride + kernel_width, j*stride : j*stride + kernel_height] * kernel[k])


    #print(f"full output {output.shape}")

    
    return output


def crossCorrelation2D(input: np.ndarray, kernel: np.ndarray, stride: int, mode:str = "valid", output_size = 0):

    inpt = input
    inp_width,inp_height = input.shape
    kernel_width, kernel_height  = kernel.shape

    w_pad = 0
    h_pad = 0

    if mode == "full":

        w_pad = kernel_width - 1
        h_pad = kernel_height - 1

        inpt = np.pad(input,(w_pad,h_pad))
        stride = 1

    out_height = ((inp_height - kernel_height + (2 * h_pad)) // stride) + 1
    out_width = ((inp_width - kernel_width + (2 * w_pad)) // stride) + 1

    if output_size != 0:

        out_height,out_width = output_size,output_size

    output = np.zeros((out_width,out_height))

    for i in range(out_height):

        for j in range(out_width):
                
                print((kernel_width,kernel_height))
                
                output[i,j] += np.sum(inpt[i*stride : i*stride + kernel_width, j*stride : j*stride + kernel_height] * kernel)

    return output


def dilate(input: np.ndarray, stride: int):

    if stride == 1:
        return input

    width, height = input.shape
    s = stride

    output_width = ((width-1) * s) + 1
    output_height = ((height-1) * s) + 1

    output = np.zeros((output_width, output_height))
    for i in range(width):
        for j in range(height):

            output[i*stride,j*stride] = input[i,j] 


    return output


def convolution3D(input: np.ndarray, kernel: np.ndarray, stride : int, mode: str = "full"):

    # To my understanding, implementations of Convlution in machine learning libraries differ on whether to rotate the kernel 180 degrees.
    # In this case ill try to be mathematically accurate and rotate the kernel for convolution. 

    # After the array needs to be dialated so that the matrix of the error with respect to the input matches the input size.
    output = []

    for i in range(kernel.shape[0]):

        output.append(dilate(kernel[i],stride))

    kernel = np.array(output)

    # print(f"Dilated kernel shape: {kernel.shape}")

    return crossCorrelation3D(input, np.rot90(np.rot90(kernel)), stride, mode)


def convolution2D(input: np.ndarray, kernel: np.ndarray, stride : int, mode: str = "full", output_size: int= 0):

    kernel = dilate(kernel,stride)

    return crossCorrelation2D(input, np.rot90(np.rot90(kernel)), stride, mode,output_size)


def pad3D(input: np.ndarray, padding: int):

    inp_depth, w,h = input.shape
    output = np.zeros((inp_depth, w + (2 * padding), h + (2 * padding)))

    for i in range(inp_depth):

        output[i] = np.pad(input[i], (padding,padding))

    return output


    
