"""Build the editable A4 submission from completed training and verification.

This creates a new notebook, never silently overwrites an edited one. Running
the notebook uses saved checkpoints and small inference/autograd checks; model
training remains in the two separate training scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Assignment4_code.ipynb")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("The notebook already exists; preserve user edits or explicitly pass --overwrite.")
    dm = json.loads((ROOT / "assignment4/diffusion_metrics.json").read_text())
    em = json.loads((ROOT / "assignment4/energy_training_metrics.json").read_text())
    for kind, metrics in (("diffusion", dm), ("energy", em)):
        checkpoint = ROOT / f"models/cifar10_{kind}.pt"
        assert metrics["train_count"] == 50000 and metrics["epochs"] > 0
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metrics["checkpoint_sha256"]
    assert "held_out_denoising" in dm, "Wait until diffusion training and evaluation finish."

    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # Assignment 4: Diffusion and Energy-Based Models

    He Xiao | UNI: hx2425  
    GitHub: [HotcatX/API-test](https://github.com/HotcatX/API-test)

    ## Part I. Model Training and API Deployment

    Two unconditional RGB image generators extend the existing FastAPI service:
    a denoising diffusion probabilistic model (DDPM) and a convolutional energy
    model (EBM). Both are trained on all 50,000 images in CIFAR-10's official
    training split, at their native resolution of 32 x 32 (Krizhevsky, 2009). Labels are ignored.
    Pixel values are normalized with `uint8 / 127.5 - 1`.

    The API loads the included trained weights on CPU. It needs no training or
    model download at startup. The earlier text, embedding, classifier, and GAN
    endpoints remain available. This notebook reads the saved artifacts and
    performs small inference and gradient checks; training is run separately.
    """)
    code("""
    from pathlib import Path
    from io import BytesIO
    import hashlib
    import json
    import os
    import sys
    import time
    import warnings

    import torch
    from PIL import Image
    from IPython.display import Markdown, display

    warnings.filterwarnings("ignore", message="Using .*httpx.*starlette.testclient.* is deprecated.*")

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                if (p / "app/diffusion_model.py").is_file())
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    torch.set_num_threads(min(2, torch.get_num_threads()))
    from app.diffusion_model import load_diffusion, sinusoidal_embedding
    from app.energy_model import load_energy_model, langevin_sample
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Project virtual environment:", Path(sys.prefix).resolve() == (ROOT / ".venv").resolve())
    """)
    md(r"""
    ### Diffusion implementation

    A small UNet predicts Gaussian noise. It uses a 64-dimensional interleaved
    sine/cosine timestep embedding, a learned time projection, GroupNorm/SiLU
    residual blocks, two downsampling stages (32 -> 16 -> 8), and two upsampling
    stages with skip connections. The final output has three channels. The
    architecture is in `app/diffusion_model.py`; the training loop is in
    `scripts/train_cifar10_diffusion.py`.

    For each minibatch, sample a timestep and noise, form
    $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$, and minimize
    $\operatorname{mean}[(\epsilon_\theta(x_t,t)-\epsilon)^2]$.
    Images and sampled noise do not require gradients; network parameters do.
    The optimizer clears old gradients before every backward pass.

    Sampling starts from Gaussian noise and uses all 100 reverse steps of a
    cosine schedule, with a clipped clean-image estimate and the DDPM posterior
    mean and variance. It uses the exponential moving average (EMA) of trained
    weights under `torch.inference_mode()` (Ho et al., 2020; Nichol & Dhariwal, 2021).
    Code indices 0 through 99 address the 100 noisy states; index 0 is the
    first noisy state, whose final reverse update produces the clean estimate.
    """)
    md(r"""
    ### Energy model and gradient control

    The energy network uses four stride-2 convolutions with channels
    16, 32, 64, and 64, SiLU activations, and a small fully connected head.
    A fixed quadratic smoothness term penalizes differences between adjacent
    pixels, so the total scalar energy per image is

    $$E_\theta(x)=f_\theta(x)+0.002
      \sum_{(i,j)\in\mathcal A}(x_i-x_j)^2,$$

    where $f_\theta$ is the trained CNN and $\mathcal A$ contains horizontal
    and vertical adjacent pixel pairs within each RGB channel. Lower energy
    is preferred. This explicit prior suppresses pixel-scale noise; it is not
    a learned measure of object realism. `app/energy_model.py` contains the
    network and sampler, and the same total energy is used in training and sampling.

    Training alternates between two phases:

    1. Freeze model parameters and run a finite Langevin-style chain on image
       tensors. Every step uses `x.detach().requires_grad_(True)` and
       `torch.autograd.grad(E(x).sum(), x)`. Pixel updates occur inside
       `torch.no_grad()`, so the graph is not retained across steps.
    2. Re-enable parameter gradients, detach the sampled negatives, clear
       optimizer gradients, and update the energy network using

    $$L=\operatorname{mean}E_\theta(x^+)
       -\operatorname{mean}E_\theta(\operatorname{stopgrad}(x^-))
       +0.1\left[\operatorname{mean}\big[E_\theta(x^+)^2\big]
                 +\operatorname{mean}\big[E_\theta(x^-)^2\big]\right].$$

    A replay buffer holds 2,048 negative images; 5% of initial states are
    refreshed with uniform noise. Training uses 20 sampling steps. API
    generation always starts from fresh uniform noise and uses 100 steps:

    $$x_{k+1}=\operatorname{clip}_{[-1,1]}
    \left(x_k-10\operatorname{clip}_{[-0.03,0.03]}\nabla_x E_\theta(x_k)
    +0.005\xi_k\right),\qquad\xi_k\sim\mathcal N(0,I).$$

    These drift, noise, clipping, and finite-chain settings implement practical
    short-run sampling; they do not establish exact equilibrium draws from
    $p(x)\propto\exp[-E_\theta(x)]$. Input-gradient tracking remains enabled
    during API sampling, while all model weights stay frozen (Lippe, 2024).
    """)
    code("""
    # Check that input-gradient sampling changes images, not model weights.
    energy = load_energy_model()
    before = {name: value.clone() for name, value in energy.state_dict().items()}
    initial = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(99)) * 2 - 1
    with torch.no_grad():
        sampled = langevin_sample(energy, initial, steps=3, noise_std=0)
    assert not sampled.requires_grad and not torch.equal(sampled, initial)
    assert all(torch.equal(before[name], value) for name, value in energy.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in energy.parameters())
    print("EBM: image pixels updated; weights unchanged; parameter gradients absent.")

    diffusion, schedule = load_diffusion()
    with torch.inference_mode():
        predicted_noise = diffusion(torch.zeros(2, 3, 32, 32), torch.tensor([0, 50]))
    assert predicted_noise.shape == (2, 3, 32, 32)
    print("DDPM noise-prediction shape:", tuple(predicted_noise.shape))
    print("Reverse diffusion steps:", schedule.timesteps)
    """)
    md("### Recorded training and diagnostics")
    code("""
    dm = json.loads((ROOT / "assignment4/diffusion_metrics.json").read_text())
    em = json.loads((ROOT / "assignment4/energy_training_metrics.json").read_text())
    dh = json.loads((ROOT / "assignment4/diffusion_history.json").read_text())
    eh = json.loads((ROOT / "assignment4/energy_training_history.json").read_text())
    for kind, metrics in (("diffusion", dm), ("energy", em)):
        actual = hashlib.sha256((ROOT / f"models/cifar10_{kind}.pt").read_bytes()).hexdigest()
        assert actual == metrics["checkpoint_sha256"]
    rows = [
        ("Training images / epoch", dm["train_count"], em["train_count"]),
        ("Epochs", dm["epochs"], em["epochs"]),
        ("Parameters", dm["parameters"], em["parameter_count"]),
        ("Batch size", dm["batch_size"], em["batch_size"]),
        ("Optimizer", dm["optimizer"], em["optimizer"]),
        ("Initial learning rate", dm["learning_rate"], em["learning_rate"]),
        ("Training seed", dm["seed"], em["seed"]),
        ("Training device", dm["training_device"], em["device"]),
        ("Training seconds", round(dm["training_seconds"], 1), round(em["elapsed_seconds"], 1)),
    ]
    display(Markdown("| Setting | DDPM | EBM |\\n|---|---:|---:|\\n" +
                     "\\n".join(f"| {key} | {a} | {b} |" for key, a, b in rows)))
    held = dm["held_out_denoising"]
    print(f"DDPM training noise MSE: {dh[0]['epsilon_mse']:.5f} -> {dh[-1]['epsilon_mse']:.5f}")
    print(f"Held-out noise MSE ({held['count']} test images): {held['epsilon_mse']:.5f}")
    print(f"Zero-noise-prediction baseline MSE: {held['zero_predictor_mse']:.5f}")
    diag = em["diagnostics"]
    print(f"EBM mean energy, fresh noise: {diag['initial_noise_mean_energy']:.5f}")
    print(f"EBM mean energy, generated: {diag['generated_mean_energy']:.5f}")
    print("Both saved checkpoint hashes verified.")
    """)
    md("""
    The final DDPM diagnostic uses the first 1,024 official test images after
    training, with fixed noise and timesteps. No test image contributes to model
    updates. Noise MSE measures denoising, and energy reduction measures the
    sampler's objective; neither is a perceptual image-quality score. FID and
    Inception Score were not measured. These compact models receive a limited
    training budget, so samples may show color and texture without clearly
    recognizable CIFAR-10 objects. Neither model accepts a class or text prompt.

    ### Generated images and API checks

    The grids below are actual CPU API responses, using the first 16 samples
    from seed 2026. They are not selected by visual quality. Each tile is a
    native 32 x 32 RGB image, with a two-pixel gap.
    """)
    code("""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    for kind in ("diffusion", "energy"):
        started = time.perf_counter()
        response = client.post(f"/generate-{kind}", json={"count": 16, "seed": 2026})
        assert response.status_code == 200, response.text
        assert response.headers["x-seed"] == "2026"
        assert response.headers["x-image-count"] == "16"
        with Image.open(BytesIO(response.content)) as picture:
            assert picture.mode == "RGB" and picture.size == (134, 134)
            display(Markdown(f"{kind.capitalize()} model: seed 2026"))
            display(picture.copy())
        (ROOT / f"assignment4/api_{kind}_seed2026.png").write_bytes(response.content)
        print(f"POST /generate-{kind}: HTTP 200, RGB 134 x 134, {time.perf_counter()-started:.2f}s")
        assert client.post(f"/generate-{kind}", json={"count": 17}).status_code == 422
    print("Invalid image counts rejected with HTTP 422.")
    """)
    md("""
    ### Run and query the service

    Build and start the included Docker deployment from the repository root:

    ```bash
    docker build -t api-test:assignment4 .
    docker run --rm -p 8000:80 api-test:assignment4
    ```

    Open `http://localhost:8000/docs` for the interactive API. The root URL
    retains the original Hello World response. Local development uses:

    ```bash
    uv sync --locked --group notebook
    uv run python -m ipykernel install --user --name api-test --display-name 'API-test (.venv)'
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run uvicorn app.main:app --port 8000
    ```

    Single-image requests return a 32 x 32 RGB PNG:

    ```bash
    curl -f 'http://localhost:8000/generate-diffusion?seed=42' -o diffusion.png
    curl -f 'http://localhost:8000/generate-energy?seed=42' -o energy.png
    ```

    A JSON request returns a PNG grid:

    ```bash
    curl -f -X POST http://localhost:8000/generate-diffusion \\
      -H 'Content-Type: application/json' \\
      -d '{"count":4,"seed":42}' -o diffusion-grid.png
    curl -f -X POST http://localhost:8000/generate-energy \\
      -H 'Content-Type: application/json' \\
      -d '{"count":4,"seed":42}' -o energy-grid.png
    ```

    `count` must be an integer from 1 to 16 (POST default: 4). The optional seed
    is an integer from 0 to 4,294,967,295. `X-Seed`, `X-Image-Count`, and `X-Model`
    identify the response. A returned random seed can be replayed using the same
    count, checkpoint, device, and runtime. Invalid inputs return HTTP 422;
    missing or incompatible trained weights return HTTP 503.
    """)
    md(f"""
    ### Reproduce training and verification

    ```bash
    uv run python scripts/train_cifar10_diffusion.py --epochs {dm['epochs']} --base-channels {dm['base_channels']}
    uv run python scripts/train_cifar10_energy.py --epochs {em['epochs']}
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q
    uv run python scripts/check_assignment4_theory.py
    uv run python scripts/smoke_api.py --base-url http://127.0.0.1:8000
    ```

    Training scripts use the project `.venv` and record settings, full-split
    epoch counts, losses, and checkpoint hashes under `assignment4/`.
    Training on different hardware may not be bitwise identical. API inference
    uses private random generators and leaves the process-wide RNG unchanged.
    """)
    code("""
    evidence = json.loads((ROOT / "assignment4/verification.json").read_text())
    display(Markdown("| Verification | Observed result |\\n|---|---|\\n" +
        "\\n".join(f"| {k} | {v} |" for k, v in evidence["checks"].items())))
    ci_path = ROOT / "assignment4/ci_verification.json"
    if ci_path.exists():
        ci = json.loads(ci_path.read_text())
        display(Markdown(f"GitHub Actions: [{ci['conclusion']}]({ci['run_url']})"))
        print("Verified runtime commit:", ci["commit"])
    """)

    # Keep the compact verification table with the training results, before
    # the sample grids, so it does not occupy a nearly empty standalone page.
    verification_cell = cells.pop()
    assert verification_cell.cell_type == "code" and "evidence = json.loads" in verification_cell.source
    sample_intro = next(i for i, cell in enumerate(cells) if "### Generated images and API checks" in cell.source)
    cells.insert(sample_intro, verification_cell)

    theory, _, references = (ROOT / "assignment4/theory_solutions.md").read_text().partition("\n## References\n")
    # The source contains worked code in fenced blocks. Execute the independent
    # assertion script below instead of treating illustrative Q5 variables as live.
    for part in re.split(r"(?=^## |^### Question )", theory, flags=re.MULTILINE):
        if part.strip():
            md(part)
    md("### Executed numerical and autograd checks")
    code("""
    from scripts.check_assignment4_theory import run_checks
    observed = run_checks()
    """)
    extra_references = [
        "Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. "
        "University of Toronto. https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf",
        "Lippe, P. (2024). Tutorial 8: Deep energy-based generative models. "
        "UvA Deep Learning Tutorials. "
        "https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html",
        "Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. "
        "Proceedings of Machine Learning Research, 139, 8162-8171. "
        "https://proceedings.mlr.press/v139/nichol21a.html",
    ]
    references = sorted(references.strip().split("\n\n") + extra_references)
    md("## References\n\n" + "\n\n".join(references))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "API-test (.venv)", "language": "python", "name": "api-test"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output)
    print(f"Created {args.output} with {len(cells)} cells")


if __name__ == "__main__":
    main()
