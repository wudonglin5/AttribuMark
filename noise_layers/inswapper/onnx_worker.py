import argparse
import json
import sys
import traceback

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def detect_keypoints(detector, image):
    height, width = image.shape[:2]
    size = max(128, int(np.ceil(max(height, width) / 32.0) * 32))
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    blob = (resized.astype(np.float32) - 127.5) / 128.0
    blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None])
    outputs = detector.run(None, {detector.get_inputs()[0].name: blob})

    best_score = 0.0
    best_points = None
    for stride_index, stride in enumerate((8, 16, 32)):
        scores = outputs[stride_index][:, 0]
        keypoint_preds = outputs[6 + stride_index]
        feature_h = size // stride
        feature_w = size // stride
        count = scores.shape[0]
        anchors_per_pixel = count // (feature_h * feature_w)

        anchor_centers = np.stack(
            np.mgrid[:feature_h, :feature_w][::-1],
            axis=-1,
        ).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape(-1, 2)
        if anchors_per_pixel > 1:
            anchor_centers = np.stack(
                [anchor_centers] * anchors_per_pixel,
                axis=1,
            ).reshape(-1, 2)

        index = int(np.argmax(scores))
        score = float(scores[index])
        if score <= best_score:
            continue
        points = keypoint_preds[index].reshape(5, 2) * stride
        points = points + anchor_centers[index]
        points[:, 0] *= width / float(size)
        points[:, 1] *= height / float(size)
        best_score = score
        best_points = points.astype(np.float32)

    if best_score < 0.3:
        return None
    return best_points


def align_face(image, keypoints, output_size=128):
    height, width = image.shape[:2]
    if keypoints is None:
        matrix = np.array(
            [
                [output_size / float(width), 0.0, 0.0],
                [0.0, output_size / float(height), 0.0],
            ],
            dtype=np.float32,
        )
    else:
        destination = ARCFACE_DST.copy()
        destination[:, 0] += 8.0
        matrix, _ = cv2.estimateAffinePartial2D(
            keypoints,
            destination,
            method=cv2.LMEDS,
        )
        if matrix is None:
            matrix = np.array(
                [
                    [output_size / float(width), 0.0, 0.0],
                    [0.0, output_size / float(height), 0.0],
                ],
                dtype=np.float32,
            )
    aligned = cv2.warpAffine(
        image,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    return aligned, matrix


def face_blob(images):
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images must be NHWC RGB, got {images.shape}")
    resized = np.stack(
        [
            cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
            for image in images
        ],
        axis=0,
    )
    if resized.dtype == np.uint8:
        resized = resized.astype(np.float32)
    else:
        resized = np.clip(resized, 0.0, 255.0).astype(np.float32)
    blob = (resized - 127.5) / 127.5
    return np.ascontiguousarray(np.transpose(blob, (0, 3, 1, 2)))


def face_embeddings(recognizer, images):
    input_name = recognizer.get_inputs()[0].name
    output_name = recognizer.get_outputs()[0].name
    result = recognizer.run([output_name], {input_name: face_blob(images)})[0]
    result = np.asarray(result, dtype=np.float32)
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norm, 1e-12)


def aligned_embeddings(recognizer, detector, images):
    aligned = []
    for image in images:
        keypoints = detect_keypoints(detector, image)
        face, _ = align_face(image, keypoints)
        aligned.append(face)
    return face_embeddings(recognizer, np.stack(aligned, axis=0))


def paste_back(target, fake, matrix):
    height, width = target.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    warped_fake = cv2.warpAffine(
        fake,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    mask = np.full(fake.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpAffine(
        mask,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    mask = (mask > 20).astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    alpha = mask.astype(np.float32)[..., None] / 255.0
    merged = alpha * warped_fake.astype(np.float32) + (
        1.0 - alpha
    ) * target.astype(np.float32)
    return np.clip(merged, 0, 255).astype(np.uint8)


def run_request(swapper, recognizer, detector, emap, request):
    input_path = request["input_npz"]
    output_path = request["output_npy"]
    with np.load(input_path) as data:
        operation = request.get("operation", "swap")
        if operation == "embed":
            images = np.asarray(data["images"])
            output = aligned_embeddings(recognizer, detector, images)
            np.save(output_path, output.astype(np.float32))
            return {
                "ok": True,
                "output": output_path,
                "shape": list(output.shape),
            }

        target = np.asarray(data["target"])
        identity = np.asarray(data["identity"])

    if target.ndim != 4 or target.shape[-1] != 3:
        raise ValueError(f"target must be NHWC RGB, got {target.shape}")
    if identity.ndim != 4 or identity.shape[-1] != 3:
        raise ValueError(f"identity must be NHWC RGB, got {identity.shape}")
    if target.shape[0] != identity.shape[0]:
        raise ValueError("target and identity batch sizes differ")

    if target.dtype != np.uint8:
        target = np.clip(target, 0, 255).astype(np.uint8)
    if identity.dtype != np.uint8:
        identity = np.clip(identity, 0, 255).astype(np.uint8)

    aligned_target = []
    target_matrices = []
    for image in target:
        keypoints = detect_keypoints(detector, image)
        face, matrix = align_face(image, keypoints)
        aligned_target.append(face)
        target_matrices.append(matrix)

    aligned_identity = []
    for image in identity:
        keypoints = detect_keypoints(detector, image)
        face, _ = align_face(image, keypoints)
        aligned_identity.append(face)

    source = face_embeddings(recognizer, np.stack(aligned_identity, axis=0))
    source = source @ emap
    source /= np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)

    outputs = []
    for index, face in enumerate(aligned_target):
        blob = np.ascontiguousarray(
            (face.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )
        result = swapper.run(
            ["output"],
            {"target": blob, "source": source[index : index + 1]},
        )[0]
        fake = np.clip(result[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        merged = paste_back(target[index], fake, target_matrices[index])
        outputs.append(merged.astype(np.float32) / 255.0)

    output = np.stack(outputs, axis=0)
    np.save(output_path, output)
    return {
        "ok": True,
        "output": output_path,
        "shape": list(output.shape),
        "min": float(output.min()),
        "max": float(output.max()),
        "mean": float(output.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--recognizer", required=True)
    parser.add_argument("--detector", required=True)
    args = parser.parse_args()

    swapper = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"],
    )
    recognizer = ort.InferenceSession(
        args.recognizer,
        providers=["CPUExecutionProvider"],
    )
    detector = ort.InferenceSession(
        args.detector,
        providers=["CPUExecutionProvider"],
    )
    model = onnx.load(args.model)
    emap = numpy_helper.to_array(model.graph.initializer[-1]).copy()
    del model
    if emap.shape != (512, 512):
        raise ValueError(f"unexpected InSwapper emap shape: {emap.shape}")

    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("close"):
                break
            response = run_request(
                swapper,
                recognizer,
                detector,
                emap,
                request,
            )
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
