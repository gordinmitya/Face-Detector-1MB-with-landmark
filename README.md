Original repository [GitHub](https://github.com/biubug6/Face-Detector-1MB-with-landmark).

Export to onnx:
```bash
python3 -m venv .venv
source .venv/bin/activate

# CMAKE_POLICY_VERSION_MINIMUM is required because of onnxoptimizer issue
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -r requirements.txt

python3 convert_to_onnx.py --network RFB --long_side 320

python -m onnxsim faceDetector.onnx faceDetector_simplified.onnx
```
