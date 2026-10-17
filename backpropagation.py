import torch

# Backpropagation is how the program calculates how much of a change a certain layer
# or individual weight has contributed to the final output.

# Mathematically...

# If you have an input, "x", which you pass through some function "a"
# and you get some output "y", which you again pass through some function "b"
# to get an output "z"
#
# Backpropagation is the process of using the chain rule of differentiation.
#
# We know that dz/dx for the above example is...
#
#     dz/dx = dz.dx * dy.dx (How does z change with a change in x?)
#
# This is used to calculate the loss function. dLoss/dx (where dx can be replaced by any other involved weight)
#
# So...
# 1. Forward pass: Computes the loss (which has to be minimized)
# 2. Computation of local gradients
# 3. Backward pass: Computes dLoss/dWeights using the chain rule, to be later utilized by the optimizer.

weight = torch.tensor(1.0)

x = torch.tensor([1.0]) # Given input
y = torch.tensor([2.0]) # Expected output

# Forward pass to compute the loss function...

y_hat = weight*x
loss = (y_hat-y)**2

print(loss)

# Backward pass to check how the weight contributes to the total loss.

loss.backward()
print(weight.grad)

# Note that the above is a dead example and does not actually run to demonstrate any of what is stated.

