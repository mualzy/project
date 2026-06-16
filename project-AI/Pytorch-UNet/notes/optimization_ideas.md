# Optimization Ideas

- Retry CUDA PyTorch install using a stable local mirror or predownloaded wheel.
- Use `--amp` after CUDA is available; AMP has no benefit for current CPU-only run.
- Keep `--num-workers 0` on Windows for small datasets; test `2` after GPU and real data are available.
- Add optional CSV logging inside training loop if a longer study is needed.
- Replace synthetic data with full Carvana data once Kaggle credentials are available.
