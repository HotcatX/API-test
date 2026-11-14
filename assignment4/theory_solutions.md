## Part II. Building Blocks of a Diffusion Model

### Question 1: Sinusoidal timestep embedding

The embedding represents the timestep using sine and cosine functions at several frequencies. For an even embedding dimension $d$, let $k=0,\ldots,d/2-1$, use a maximum-period parameter $M=10000$, and define

$$
\omega_k=M^{-2k/d}.
$$

Using zero-based coordinates and interleaved sine/cosine pairs,

$$
e_{2k}(t)=\sin(t\omega_k),
\qquad
e_{2k+1}(t)=\cos(t\omega_k).
$$

Equivalently, the $i$-th coordinate is

$$
e_i(t)=
\begin{cases}
\sin\left(t/M^{i/d}\right), & i\text{ even},\\
\cos\left(t/M^{(i-1)/d}\right), & i\text{ odd}.
\end{cases}
$$

Angles are measured in radians. Higher-frequency coordinates change more rapidly with $t$, while lower-frequency coordinates change more slowly. This follows the sinusoidal positional-encoding construction of Vaswani et al. (2017). Here, $M$ is the conventional frequency-scale parameter; the actual sinusoid period is $2\pi/\omega_k$.

### Question 2: Eight-dimensional embedding at $t=1$

With $d=8$, the four frequency denominators are

$$
10000^{0/8}=1,\quad
10000^{2/8}=10,\quad
10000^{4/8}=100,\quad
10000^{6/8}=1000.
$$

Therefore, the angles are $1$, $0.1$, $0.01$, and $0.001$, and the vector is

$$
e(1)=
[\sin(1),\cos(1),\sin(0.1),\cos(0.1),
\sin(0.01),\cos(0.01),\sin(0.001),\cos(0.001)].
$$

Numerically, to nine decimal places,

```text
[0.841470985, 0.540302306, 0.099833417, 0.995004165,
 0.009999833, 0.999950000, 0.001000000, 0.999999500]
```

The ordering follows Question 1. An implementation that concatenates all cosines followed by all sines has the same values in a different order:

```text
[0.540302306, 0.995004165, 0.999950000, 0.999999500,
 0.841470985, 0.099833417, 0.009999833, 0.001000000]
```

### Question 3: Relation to transformer positional encoding

Both methods map a scalar index to a vector of sine and cosine values at different frequencies. In the original Transformer, the scalar is a token's position in a sequence; the encoding is added to token embeddings so the model can use sequence order (Vaswani et al., 2017). In a diffusion model, the scalar is the diffusion timestep or noise level; the embedding conditions the denoising network so it knows how much noise to remove. A timestep embedding may be passed through a learned projection and supplied to multiple UNet blocks (Ho et al., 2020). The key difference is what the index describes: token location versus denoising stage.

### Question 4: Bottleneck spatial resolution

Assuming each stride-2 block halves the height and width, the resolutions are

$$
64\times64\longrightarrow32\times32
\longrightarrow16\times16\longrightarrow8\times8.
$$

Thus,

$$
H_{\mathrm{bottleneck}}=W_{\mathrm{bottleneck}}
=\frac{64}{2^3}=8.
$$

The bottleneck spatial resolution is $8\times8$. For example, kernel size 3, stride 2, and padding 1 give this halving. The channel count depends on the architecture and is not specified by the question.

### Question 5: UNet output and diffusion loss

For the noise-prediction formulation, the UNet outputs an estimate $\epsilon_\theta(x_t,t)$ of the noise added to the image. Its output has the same shape as $x_t$; for RGB images, this is $(B,3,H,W)$.

Define $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$. To create a training input, sample a clean image $x_0$, a timestep $t$, and Gaussian noise $\epsilon\sim\mathcal N(0,I)$:

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+
\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

Train the UNet to predict the sampled $\epsilon$ using mean squared error:

$$
\mathcal L_{\mathrm{MSE}}
=\frac{1}{BCHW}\sum_{b,c,h,w}
\left[\epsilon_{bchw}-
\epsilon_\theta(x_t,t)_{bchw}\right]^2.
$$

This is the simplified DDPM noise-prediction objective, with an elementwise mean used in code (Ho et al., 2020):

```python
noise = torch.randn_like(x0)
alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
xt = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise
predicted_noise = model(xt, t)
loss = torch.nn.functional.mse_loss(predicted_noise, noise)
```

The prediction is noise, rather than the clean image directly. If needed, an estimate of the clean image is

$$
\hat x_0=
\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

## Part III. Building Blocks of an Energy Model

### Question 6: Basic gradient calculations

(a) The derivative of $y=x^2+3x$ is

$$
\frac{dy}{dx}=2x+3.
$$

At $x=2$, this gives $2(2)+3=7$. Running the supplied code prints

```text
x.grad = tensor([7.])
```

(b) If $x$ is created with `requires_grad=False`, none of the tensors in this particular expression require gradients. Thus, `y` has no autograd history, and `y.backward()` raises

```text
RuntimeError: element 0 of tensors does not require grad
and does not have a grad_fn
```

`x.grad` remains `None`. This conclusion concerns the supplied expression: if another input required gradients, an operation could still be tracked for that input.

(c) The default for `torch.tensor` is `requires_grad=False`. Gradients are therefore not tracked for the newly created tensor unless explicitly enabled (PyTorch Contributors, n.d.-a).

### Question 7: Introduce weights

(a) In the original code, `w = torch.tensor([1.0, 3.0])` uses the default `requires_grad=False`, so

```text
x.grad = tensor([7.])
w.grad = None
```

The values of `w` affect the calculation of $y$, but PyTorch has not been instructed to compute gradients with respect to `w`.

(b) For $y=w_0x^2+w_1x$, the partial derivatives are

$$
\frac{\partial y}{\partial w_0}=x^2,
\qquad
\frac{\partial y}{\partial w_1}=x.
$$

At $x=2$, they are $4$ and $2$. Enable gradient tracking before constructing the forward computation:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print("x.grad =", x.grad)
print("w.grad =", w.grad)
```

Output:

```text
x.grad = tensor([7.])
w.grad = tensor([4., 2.])
```

If only weight gradients are needed, `x` can instead have `requires_grad=False`; `w.grad` is still `[4., 2.]`.

(c) Omitting the flag again gives `requires_grad=False`; using variable coefficients does not change this default (PyTorch Contributors, n.d.-a).

### Question 8: Breaking the graph

`z = y.detach()` removes the differentiable connection to `y` and `x`. The returned tensor does not require gradients (PyTorch Contributors, n.d.-c). Consequently, `w = z * 2` also has no autograd history, and `w.backward()` raises the same no-gradient-history error as in Question 6.

Keep the variable `z` but preserve its connection to `y`:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone()  # A differentiable copy; z = y would also work.
w = z * 2
w.backward()
print("x.grad =", x.grad)
```

By the chain rule,

$$
\frac{dw}{dx}
=\frac{dw}{dz}\frac{dz}{dy}\frac{dy}{dx}
=2\cdot1\cdot3=6.
$$

The output is `x.grad = tensor([6.])`.

Using `z = y.detach().requires_grad_(True)` would make `z` a new leaf, but would not reconnect it to `x`: `z.grad` would be `[2.]`, while `x.grad` would remain `None`.

### Question 9: Gradient accumulation

The first backward pass adds $dy_1/dx=2$ to the empty gradient buffer. The second pass adds $dy_2/dx=3$ to the existing value:

$$
x.\mathrm{grad}:\quad 2\longrightarrow2+3=5.
$$

The supplied code prints

```text
After first backward: x.grad = tensor([2.])
After second backward: x.grad = tensor([5.])
```

PyTorch accumulates gradients in leaf tensors instead of replacing them on every backward pass (PyTorch Contributors, n.d.-b). To compute independent gradients, reset the buffer before the next backward pass:

```python
x = torch.tensor([1.0], requires_grad=True)
(x * 2).backward()
print("After first backward: x.grad =", x.grad)

x.grad = None  # Alternatively: x.grad.zero_()
(x * 3).backward()
print("After reset and second backward: x.grad =", x.grad)
```

The two outputs are `[2.]` and `[3.]`. In a training loop, the usual equivalent is `optimizer.zero_grad(set_to_none=True)` before `loss.backward()`. Accumulation can also be intentional, such as when combining gradients from several minibatches.

## References

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems, 33*, 6840-6851. https://arxiv.org/abs/2006.11239

PyTorch Contributors. (n.d.-a). *torch.tensor*. PyTorch 2.8 documentation. https://docs.pytorch.org/docs/2.8/generated/torch.tensor.html

PyTorch Contributors. (n.d.-b). *torch.Tensor.backward*. PyTorch 2.8 documentation. https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.backward.html

PyTorch Contributors. (n.d.-c). *torch.Tensor.detach*. PyTorch 2.8 documentation. https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.detach.html

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1706.03762
