# Assignment 3: Theory - Building Blocks of GANs

He Xiao (hx2425)

## Question 1

Given an 8 × 8 input, find the output size of a transposed convolution with a 4 × 4 kernel, stride 2, padding 1, and output padding 1. Show the calculation.

For one spatial dimension, with dilation equal to 1,

$$
O=(I-1)S-2P+K+OP.
$$

Substituting the given values,

$$
O=(8-1)(2)-2(1)+4+1=14-2+4+1=17.
$$

The same calculation applies to height and width, so the output spatial size is 17 × 17. The number of output channels depends on the number of filters, which is not specified.

## Question 2

How does the output size change if the stride increases from 2 to 3 while everything else stays fixed?

Using the input, kernel, padding, and output padding from Question 1,

$$
O_{S=3}=(8-1)(3)-2(1)+4+1=21-2+4+1=24.
$$

The output changes from 17 × 17 to 24 × 24. Each spatial dimension increases by

$$
\Delta O=(I-1)\Delta S=(8-1)(3-2)=7.
$$

## Question 3

Write the output-size formula for a 2D transposed convolution in terms of input size I, kernel size K, stride S, padding P, and output padding OP.

Assuming dilation 1, the formula for each spatial dimension is

$$
O=(I-1)S-2P+K+OP.
$$

For a rectangular input or different height/width settings, apply it separately:

$$
H_{out}=(H_{in}-1)S_h-2P_h+K_h+OP_h,
$$

$$
W_{out}=(W_{in}-1)S_w-2P_w+K_w+OP_w.
$$

More generally, with dilation D, replace K by the effective kernel size D(K - 1) + 1:

$$
O=(I-1)S-2P+D(K-1)+OP+1.
$$

This is the shape convention used by [PyTorch ConvTranspose2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.ConvTranspose2d.html).

## Question 4

Give one kernel-size and stride configuration that upsamples a 16 × 16 feature map to 32 × 32 using a transposed convolution with no padding.

Choose a 2 × 2 kernel and stride 2, with padding 0, output padding 0, and dilation 1. Then

$$
O=(16-1)(2)-2(0)+2+0=30+2=32.
$$

Therefore, kernel_size=2 and stride=2 give the required 32 × 32 output.

## Question 5

For the mini-batch [6, 8, 10, 6], compute BatchNorm's normalized output without the learned scale and shift parameters gamma and beta.

The batch mean is

$$
\mu_B=\frac{6+8+10+6}{4}=\frac{30}{4}=7.5.
$$

Batch normalization uses the mini-batch variance with denominator 4:

$$
\sigma_B^2=\frac{(6-7.5)^2+(8-7.5)^2+(10-7.5)^2+(6-7.5)^2}{4}
$$

$$
=\frac{2.25+0.25+6.25+2.25}{4}=\frac{11}{4}=2.75.
$$

Thus the standard deviation is

$$
\sigma_B=\sqrt{2.75}=\frac{\sqrt{11}}{2}\approx1.658312.
$$

For the requested ideal zero-mean, unit-variance calculation, take epsilon to be 0. Normalize each value using

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sigma_B}.
$$

The normalized output, in the original order, is

$$
\hat{x}=\left[\frac{-3}{\sqrt{11}},\frac{1}{\sqrt{11}},\frac{5}{\sqrt{11}},\frac{-3}{\sqrt{11}}\right]
$$

$$
\approx[-0.904534,\;0.301511,\;1.507557,\;-0.904534].
$$

The normalized mean is (-3 + 1 + 5 - 3)/(4 sqrt(11)) = 0. Its batch variance is

$$
\frac{1}{4}\left(\frac{9+1+25+9}{11}\right)=\frac{44}{44}=1.
$$

In practice, BatchNorm divides by sqrt(variance + epsilon) for numerical stability, so a positive epsilon makes the variance slightly less than 1. PyTorch uses the denominator-N variance for the training forward pass; its running-variance update uses an unbiased estimate instead. These are distinct operations. See [PyTorch BatchNorm2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BatchNorm2d.html).

## Question 6

What is the key mathematical difference between ReLU and LeakyReLU? Give their formulas.

ReLU maps every negative input to zero:

$$
\operatorname{ReLU}(x)=\max(0,x)=
\begin{cases}
x, & x\geq0,\\
0, & x<0.
\end{cases}
$$

LeakyReLU uses a small positive slope alpha on the negative side:

$$
\operatorname{LeakyReLU}(x)=
\begin{cases}
x, & x\geq0,\\
\alpha x, & x<0,
\end{cases}
\qquad 0<\alpha<1.
$$

The key difference is the slope for x < 0: ReLU's slope is 0, while LeakyReLU's slope is alpha. The specified discriminator uses alpha = 0.2, so an input of -5 produces 0 under ReLU and -1 under LeakyReLU. The formula follows [PyTorch LeakyReLU](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LeakyReLU.html).

## Question 7

Why might LeakyReLU be preferred over ReLU in deep networks?

For negative pre-activation x, ReLU has derivative 0. By the chain rule,

$$
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}.
$$

If a ReLU unit receives negative inputs for all relevant training examples, its activation derivative blocks the gradient through that unit. Its incoming weights may then stop receiving useful updates: the dying-ReLU problem.

LeakyReLU has derivative alpha > 0 for x < 0, so

$$
\frac{\partial L}{\partial x}=\alpha\frac{\partial L}{\partial y}.
$$

A nonzero upstream gradient can therefore still reach the unit's incoming weights when its input is negative. This reduces the risk of permanently inactive units and can help optimization. It does not guarantee better performance or eliminate all vanishing-gradient problems; the result still depends on the model, initialization, data, and training procedure.
