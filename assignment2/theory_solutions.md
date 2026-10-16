## Part III. Theory: Arithmetic of CNNs

For these calculations, let $n$ be the input height or width, $k$ the kernel size, $s$ the stride, and $p$ the padding on each side. With dilation $1$, the output size along one spatial dimension is

$$
n_{\mathrm{out}}=\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1.
$$

The same calculation applies independently to height and width. A convolution's number of output channels equals its number of filters. Shapes in the answers use height-width-channels notation; PyTorch tensors use batch-channels-height-width notation. [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

### Question 1

Given a $32\times32\times3$ input, eight $5\times5$ filters, stride $1$, and no padding, what is the output size?

Substitute $n=32$, $k=5$, $s=1$, and $p=0$:

$$
H_{\mathrm{out}}=W_{\mathrm{out}}
=\left\lfloor\frac{32+2(0)-5}{1}\right\rfloor+1
=27+1=28.
$$

Each filter combines information from all three input channels and produces one feature map. Eight filters produce eight output channels, so the output shape is

$$
\boxed{28\times28\times8}.
$$

For a batch of $N$ images, the PyTorch output shape is $(N,8,28,28)$.

### Question 2

How does the output size in Question 1 change if padding is changed to “same”?

With “same” padding and stride $1$, the output keeps the input's spatial dimensions. A $5\times5$ kernel requires padding of

$$
p=\frac{k-1}{2}=\frac{5-1}{2}=2
$$

on each side. Therefore,

$$
H_{\mathrm{out}}=W_{\mathrm{out}}
=\left\lfloor\frac{32+2(2)-5}{1}\right\rfloor+1
=31+1=32.
$$

There are still eight filters, so the output shape is

$$
\boxed{32\times32\times8}.
$$

### Question 3

What is the output spatial size for a $64\times64$ input, a $3\times3$ filter, stride $2$, and no padding?

Substitute $n=64$, $k=3$, $s=2$, and $p=0$:

$$
\begin{aligned}
H_{\mathrm{out}}=W_{\mathrm{out}}
&=\left\lfloor\frac{64+2(0)-3}{2}\right\rfloor+1\\
&=\left\lfloor30.5\right\rfloor+1\\
&=30+1=31.
\end{aligned}
$$

The fractional value is rounded down because only positions where the entire kernel fits inside the input are valid. The output spatial size is

$$
\boxed{31\times31}.
$$

The number of output channels would depend on the number of filters, which is not specified.

### Question 4

What is the output size after $2\times2$ max pooling with stride $2$ on a $16\times16$ feature map?

Using $n=16$, $k=2$, $s=2$, and the default of no padding,

$$
H_{\mathrm{out}}=W_{\mathrm{out}}
=\left\lfloor\frac{16-2}{2}\right\rfloor+1
=7+1=8.
$$

Each non-overlapping $2\times2$ region is replaced by its maximum. Both spatial dimensions are halved, giving

$$
\boxed{8\times8}.
$$

Pooling preserves the number of channels. If there are $C$ input channels, the complete output shape is $8\times8\times C$.

### Question 5

What is the output shape after two successive convolutions on a $128\times128$ image, each using a $3\times3$ kernel, stride $1$, and “same” padding?

Each convolution uses padding

$$
p=\frac{3-1}{2}=1.
$$

After the first convolution,

$$
H_1=W_1
=\left\lfloor\frac{128+2(1)-3}{1}\right\rfloor+1
=127+1=128.
$$

After the second convolution,

$$
H_2=W_2
=\left\lfloor\frac{128+2(1)-3}{1}\right\rfloor+1
=127+1=128.
$$

Therefore, the final spatial shape is

$$
\boxed{128\times128}.
$$

The number of filters is not provided, so the complete output shape cannot be determined numerically. If the second convolution has $C_2$ filters, the complete shape is $128\times128\times C_2$.

### Question 6

What happens if `model.train()` is removed before the training loop?

Calling `model.train()` puts the model and its submodules into training mode. Removing it does not necessarily prevent training.

A newly created PyTorch model starts in training mode by default. If it has never been switched to evaluation mode, removing `model.train()` leaves it in training mode and does not change its behavior. [PyTorch modules documentation](https://docs.pytorch.org/docs/stable/notes/modules.html)

If `model.eval()` was called earlier, the model remains in evaluation mode unless training mode is restored. This matters for layers such as Dropout and BatchNorm: Dropout remains inactive, and BatchNorm uses its stored running statistics instead of its training behavior. [PyTorch Module documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

The CNN in this assignment contains only convolution, ReLU, max-pooling, flattening, and linear layers. These layers have the same forward behavior in both modes, so removing `model.train()` does not change this particular network's forward computation.

Training mode is separate from gradient tracking. `model.eval()` does not disable gradients, and `model.train()` does not override `torch.no_grad()` or `torch.inference_mode()`. Parameter updates still require gradient computation, `loss.backward()`, and an optimizer step. Explicitly calling `model.train()` before training remains good practice. [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/notes/autograd.html)
