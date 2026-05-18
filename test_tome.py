"""
Quick single-image test for SmolVLM with/without ToMe.

Usage:
    python test_tome.py
    python test_tome.py --image path/to/image.jpg --question "Describe this image"
    python test_tome.py --image path/to/image.jpg --r 60
"""

import argparse
from vlmeval.vlm.smolvlm import SmolVLM

MODEL_PATH = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEFAULT_IMAGE = "assets/Horse.png"
DEFAULT_QUESTION = "Describe this image?"


def run(image, question, r):
    msg = [
        {"type": "image", "value": image},
        {"type": "text",  "value": question},
    ]

    print(f"\nImage   : {image}")
    print(f"Question: {question}\n")

    for use_tome, label in [(False, "Baseline (r=0)"), (True, f"ToMe    (r={r})")]:
        m = SmolVLM(model_path=MODEL_PATH, use_tome=use_tome, r=r)
        answer = m.generate_inner(msg)
        print(f"  {label}: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    default=DEFAULT_IMAGE,    help="Path to input image")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to ask")
    parser.add_argument("--r",        default=60, type=int,     help="ToMe r value (tokens merged per layer)")
    args = parser.parse_args()

    run(args.image, args.question, args.r)
