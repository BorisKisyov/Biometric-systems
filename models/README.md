This folder stores the pretrained FingerFlow model files used by the demo.

Expected files:
- CoarseNet.h5
- FineNet.h5
- ClassifyNet.h5
- yolo-kernel_best.weights
- VerifyNet10.h5
- VerifyNet14.h5
- VerifyNet20.h5
- VerifyNet24.h5
- VerifyNet30.h5

They are downloaded automatically when you start the Docker container with
AUTO_DOWNLOAD_MODELS=1, or manually through:

- `python -m app.download_models`
- `powershell -ExecutionPolicy Bypass -File .\scripts\setup-local.ps1`

These binary weight files are intentionally not committed to GitHub.
