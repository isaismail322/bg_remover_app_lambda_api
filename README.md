# U²-Net Background Removal Lambda

## Overview
- Handler: [`lambda_function.py:1`](lambda_function.py:1) implements a Lambda entry point that accepts a JSON payload containing a base64-encoded image, runs the ONNX U²-Net model via the [`onnxruntime`](requirements.txt:1) dependency bundle, and returns both the input rendered in RGBA and the alpha mask as base64 strings.
- Dependencies: listed in [`requirements.txt:1`](requirements.txt:1) so the runtime can load `onnxruntime`, `numpy`, and `Pillow`.

## API Contract
- Expect an HTTP POST (API Gateway proxy) where the JSON body looks like the example in [`event.json:1`](event.json:1). The `image` key must hold a base64-encoded RGB image. The Lambda returns a JSON payload with a `result` map containing `image` (RGBA) and `mask` (grayscale) base64 strings from the U²-Net output.
- Error responses include a 400 for validation issues and 500 for unexpected failures.

## Lambda Layer + ONNX Model
1. Build the ONNX layer artifact:
   ```sh
   mkdir -p layer/opt/onnx-models
   cp path/to/u2net.onnx layer/opt/onnx-models/u2net.onnx
   (cd layer && zip -r ../u2net-layer.zip opt)
   ```
2. Upload `u2net-layer.zip` as a Lambda layer targeting the same Python runtime as your function.
3. Configure the function to mount the layer; this will expose `/opt/onnx-models/u2net.onnx` at runtime.
4. Optionally set `ONNX_MODEL_PATH` to override the default path (`/opt/onnx-models/u2net.onnx`), which is useful if your layer places the model elsewhere.

Download link https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

## Lambda Deployment Checklist
1. Install dependencies into the deployment package (or build in a container matching the Lambda runtime):
   ```sh
   python -m pip install -r requirements.txt -t package
   cp lambda_function.py package/
   cd package && zip -r ../lambda.zip .
   ```
2. Deploy `lambda.zip` and attach the ONNX layer. Increase the function memory/timeout to accommodate ONNX inference (e.g., 512 MB, 30 seconds).
3. Point an API Gateway HTTP API at `lambda_handler` and pass through binary/base64 bodies if you send base64 directly; otherwise encode back to base64 before invoking.
4. Log output goes to CloudWatch; the handler already logs warnings and stack traces when exceptions occur.

## Building a Local U²-Net Model

- Use the helper script to download the pretrained U²-Net variant (default `u2net`) via `torch.hub`, optionally override the checkpoint, and export it to ONNX:
  ```sh
  python build_u2net.py --output path/to/u2net.onnx
  ```
- Choose a specific variant through `--variant {u2net,u2netp}`, change the dummy tensor size with `--size`, and target the desired ONNX opset with `--opset`.
- Provide `--checkpoint` after `--no-pretrained` to load a custom state dict instead of the hub weights, and use `--force-reload` to refresh the download.
- The script depends on [`build_u2net.py:1`](build_u2net.py:1) and requires [`requirements.txt:1`](requirements.txt:1) for `torch` alongside the existing runtime packages.

## Sample Event
See the template in [`event.json:1`](event.json:1) for the expected payload structure. Replace the placeholder with a real base64-encoded image when testing.
