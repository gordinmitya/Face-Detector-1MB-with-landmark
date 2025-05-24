from dataclasses import dataclass
import argparse

import torch

from data import cfg_mnet, cfg_slim, cfg_rfb
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB


@dataclass(frozen=True)
class ModelOptions:
    model: type
    cfg: dict
    weights: str


options = {
    "mobile0.25": ModelOptions(
        model=RetinaFace, cfg=cfg_mnet, weights="./weights/mobilenet0.25_Final.pth"
    ),
    "RFB": ModelOptions(model=RFB, cfg=cfg_rfb, weights="./weights/RBF_Final.pth"),
    "slim": ModelOptions(model=Slim, cfg=cfg_slim, weights="./weights/slim_Final.pth"),
}

parser = argparse.ArgumentParser(description="Face-Detector-1MB-with-landmark")
parser.add_argument(
    "--network", default="RFB", help="Backbone network mobile0.25 or slim or RFB"
)
parser.add_argument(
    "--long_side",
    default=320,
    help="when origin_size is false, long_side is scaled size(320 or 640 for long side)",
)

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print("Missing keys:{}".format(len(missing_keys)))
    print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    print("remove prefix '{}'".format(prefix))

    def f(x) -> str:
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    if args.network not in options:
        print("Don't support network!")
        exit(0)

    option = options[args.network]
    cfg = option.cfg
    net = option.model(cfg=cfg, phase="test")

    # load weight
    net = load_model(net, option.weights, True)
    net.eval()
    print("Finished loading model!")
    print(net)
    device = torch.device("cpu")
    net = net.to(device)

    ##################export###############
    output_onnx = "faceDetector.onnx"
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input0"]
    output_names = ["output0"]
    side = int(args.long_side)
    inputs = torch.randn(1, 3, side, side).to(device)
    torch_out = torch.onnx.export(
        net,
        inputs,
        output_onnx,
        export_params=True,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    print("==> ONNX export finished!")
    ##################end###############
