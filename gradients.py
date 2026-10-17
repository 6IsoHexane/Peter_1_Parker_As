import torch

# Gradients are a way of tracking the changes that weights make to our original tensor.
# Mathematically, it is the inverse of the chain rule.
# i.e for a tensor x, if you run it through some function "a" with some weight "w1"
# and then run it through another function "b" with a weight "w2"
# The gradients package calculates the changes each of the functions made individually,
# And also the total change all the functions have made.

# During the forward pass, the neural network makes a prediction...
# During the backward pass, it figures out how each parameter or layer of weight...
# Contributes to the value of the loss function.
# After the forward and backward pass, during the optimization step, the model corrects each parameter accordingly.

# When later, the backpropagation command is invoked, the values of the whatever final
# answer we're asking to backpropagate MUST be a scalar.
# To do so, we have to turn the matrix into a scalar, by either taking the mean or the sum of all matrix elements.

# Note that the requires_grad function only tells the program to keep track of the gradient.
# It does not calculate dy/dx just yet.

#_________________________________________________________________________________________________________________________________________________________

x = torch.randn(3, requires_grad=True) # The requires_grad condition is a must to track the tensor gradients as they change.
print(x)

y = x+2
print(y) # Output will show: "grad_fn=<AddBackward0>"
#X and 2 are inputs, y is the output, (also a function)...

# When you have the requires_grad condition set to true, Pytorch calculates dy/dx.
# i.e, "how much does y change when you change x?"

z = x*2
print(z) # Output will show "grad_fn=<MulBackward0>"

# To later stop tracking the gradient...

x.requires_grad_(False) #OR
x.detach_()