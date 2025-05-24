Original repository [GitHub](https://github.com/biubug6/Face-Detector-1MB-with-landmark).

Export to onnx:
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python3 convert_to_onnx.py --network RFB --long_side 320
```
