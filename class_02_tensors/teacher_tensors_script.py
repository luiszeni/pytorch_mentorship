import torch
import numpy as np


#####################################################################################
#Step 1: Creating Tensors**
#####################################################################################

# Creating a tensor from a list
data_list = [1, 2, 3]
tensor_from_list = torch.tensor(data_list)
print("Tensor from list:", tensor_from_list)


# Creating a tensor from a NumPy array
data_numpy = np.array([4, 5, 6])
tensor_from_numpy = torch.tensor(data_numpy)
print("Tensor from NumPy array:", tensor_from_numpy)


# Creating a tensor of zeros and ones with specified shape
zeros_tensor = torch.zeros(2, 3)  # 2 rows, 3 columns
ones_tensor = torch.ones((4, 2, 3, 10))   # 3 rows, 2 columns


print("Zeros tensor:\n", zeros_tensor)
print("Ones tensor:\n", ones_tensor)


# Creating a tensor with random values between 0 and 1
random_tensor = torch.rand(3, 3)  # 3 rows, 3 columns
print("Random tensor:\n", random_tensor)



# Creating tensors with specific data types
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int)
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)

# Creating tensors on GPU
cuda_tensor = torch.tensor([1, 2, 3], device='cuda')

print("Integer tensor:", int_tensor)
print("Float tensor:", float_tensor)
print("CUDA tensor:", cuda_tensor)


# Creating a tensor on CPU
cpu_tensor = torch.tensor([1.0, 2.0, 3.0])


# Moving a tensor to GPU (if available)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to('cuda')
    print("Tensor on GPU:", gpu_tensor)
else:
    print("GPU not available.")


cpu_again = gpu_tensor.to('cpu')
print("Tensor on CPU again:", cpu_again)

gpu_tensor.cuda().cpu()


#####################################################################################
#Step 3: Tensor Properties and Attributes**
#####################################################################################

# Creating a tensor
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

# Discuss tensor properties
print("Tensor Properties:")
print("- Shape:", tensor.shape)
print("- Data Type:", tensor.dtype)
print("- Device:", tensor.device)
print("- Number of Dimensions (Rank):", tensor.ndim)
print()


#####################################################################################
#Step 4: Basic Operations**
#####################################################################################

# Creating tensors
tensor1 = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])

tensor2 = torch.tensor([[2, 3, 4],
                        [5, 6, 7]])

# Basic element-wise operations
print("Basic Element-Wise Operations:")
print("- Addition:\n", tensor1 + tensor2)
print("- Subtraction:\n", tensor1 - tensor2)
print("- Multiplication:\n", tensor1 * tensor2)
print("- Division:\n", tensor1 / tensor2)
print()

# Broadcasting for element-wise operations
print("Broadcasting:")
scalar = torch.tensor(10)
result = tensor1 * scalar
print("Adding scalar to tensor:\n", result)
print()

# Introducing common mathematical functions
print("Common Mathematical Functions:")
exponential = torch.exp(tensor1)
square_root = torch.sqrt(tensor2)
print("- Exponential:\n", exponential)
print("- Square Root:\n", square_root)


#####################################################################################
#Step 5: Indexing and Slicing**
#####################################################################################

# Creating a tensor
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Indexing and slicing
print("Indexing and Slicing:")
print("- Element at [1, 1]:", tensor[1, 1])
print("- First row:", tensor[0])
print("- Second column:", tensor[:, 1])
print("- Sub-tensor (top-left 2x2):\n", tensor[1:3, 1:3])
print()

# Negative indices and strides
print("Negative Indices and Strides:")
print("- Last element:", tensor[-1, -2])
print("- Every second row:\n", tensor[::2, ::2])


#####################################################################################
#Step 6: Reshaping Tensors**
#####################################################################################

# Creating a tensor
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])


# Reshaping with .view()
reshaped_view = tensor.view(3, 2)
print("Reshaped with .view():\n", reshaped_view)

# Reshaping with .reshape()
reshaped_reshape = tensor.reshape(1, 6)
print("Reshaped with .reshape():\n", reshaped_reshape)
print()

# In-place and out-of-place reshaping
print("In-Place vs. Out-of-Place Reshaping:")
original_tensor = torch.tensor([[1, 2, 3],
                                [4, 5, 6]])

# In-place reshaping
in_place_reshaped = original_tensor.view(3, -1)
print("In-Place Reshaped:\n", in_place_reshaped)

# Out-of-place reshaping
out_of_place_reshaped = original_tensor.reshape(-1, 3)
print("Out-of-Place Reshaped:\n", out_of_place_reshaped)

print("Original Tensor (Unchanged):\n", original_tensor)


#####################################################################################
#Step 7: Reduction Operations**
#####################################################################################

# Creating a tensor
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

print(tensor.dtype)
# Reduction operations
print("Reduction Operations:")
print("- Sum:", tensor.sum())
print("- Mean:", tensor.float().mean())
print("- Standard Deviation:", tensor.float().std())
print("- Maximum:", tensor.max())
print()

# Reduction along specific dimensions
print("Reduction Along Specific Dimensions:")
tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]).float()

# Sum along rows (dim=0)
sum_along_rows = tensor_2d.sum(dim=1)
print("Sum along rows (dim=0):\n", sum_along_rows)

# Mean along columns (dim=1)
mean_along_columns = tensor_2d.mean(dim=1)
print("Mean along columns (dim=1):\n", mean_along_columns)


#####################################################################################
#Step 8: Matrix Operations**
#####################################################################################

# Creating tensors
matrix1 = torch.tensor([[1, 2],
                        [3, 4]])

matrix2 = torch.tensor([[5, 6],
                        [7, 8]])

# Matrix multiplication
print("Matrix Multiplication:")
matrix_product = torch.mm(matrix1, matrix2)
print("Matrix Product (torch.mm()):\n", matrix_product)

# Matrix multiplication with torch.matmul()
matrix_product_matmul = torch.matmul(matrix1, matrix2)
print("Matrix Product (torch.matmul()):\n", matrix_product_matmul)
print()

# Transposition
print("Transposition:")
matrix_transpose = matrix1.T
print("Transposed Matrix:\n", matrix_transpose)
print()

# Determinant
print("Determinant:")
matrix3 = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]).float()
determinant = torch.det(matrix3)
print("Determinant:\n", determinant)



#####################################################################################
#Step 9: Gradients and Autograd**
#####################################################################################

# Creating tensors with requires_grad=True
tensor1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
tensor2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Illustrating automatic differentiation
print("Illustrating Automatic Differentiation:")
result = tensor1 + tensor2
print("Result tensor:", result)

# Computing gradients using .backward()
print("Computing Gradients using .backward():")
loss = result.sum()
loss.backward()

# Gradients
print("Gradients:")
print("Gradient of tensor1:", tensor1.grad)
print("Gradient of tensor2:", tensor2.grad)




#####################################################################################
#Step 10: Conversion to/from NumPy**
#####################################################################################

# Converting tensors to NumPy arrays
tensor = torch.tensor([1.0, 2.0, 3.0])
numpy_array = tensor.numpy()

print("Tensor to NumPy:")
print("Original Tensor:", tensor)
print("Converted NumPy Array:", numpy_array)
print()

# Converting NumPy arrays to tensors
numpy_arr = np.array([4.0, 5.0, 6.0])
tensor_from_numpy = torch.from_numpy(numpy_arr)

print("NumPy to Tensor:")
print("Original NumPy Array:", numpy_arr)
print("Converted Tensor:", tensor_from_numpy)

#####################################################################################
#Step 11: Concatenation and Stacking**
#####################################################################################

# Creating tensors
tensor1 = torch.tensor([[1, 2],
                        [3, 4]])

tensor2 = torch.tensor([[5, 6],
                        [7, 8]])

# Concatenation using torch.cat()
concatenated_rows    = torch.cat((tensor1, tensor2), dim=0)
concatenated_columns = torch.cat((tensor1, tensor2), dim=1)

print("Concatenation using torch.cat():")
print("Concatenated Rows:\n", concatenated_rows)
print("Concatenated Columns:\n", concatenated_columns)
print()

# Stacking using torch.stack()
stacked_tensors = torch.stack((tensor1, tensor2, tensor1, tensor2))

print("Stacking using torch.stack():")
print("Stacked Tensors:\n", stacked_tensors.shape)