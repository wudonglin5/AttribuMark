import argparse
import json
import time

import numpy as np


def build_gf32():
    exp = [0] * 62
    log = [0] * 32
    value = 1
    for index in range(31):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x20:
            value ^= 0x25
        value &= 0x1F
    for index in range(31, 62):
        exp[index] = exp[index - 31]
    return exp, log


def gf_mul(left, right, exp, log):
    if left == 0 or right == 0:
        return 0
    return exp[log[left] + log[right]]


def minimal_polynomial(root, exp, log):
    conjugates = []
    value = root
    for _ in range(5):
        if value not in conjugates:
            conjugates.append(value)
        value = gf_mul(value, value, exp, log)
    poly = [1]
    for conjugate in conjugates:
        updated = [0] * (len(poly) + 1)
        for degree, coefficient in enumerate(poly):
            updated[degree + 1] ^= coefficient
            updated[degree] ^= gf_mul(coefficient, conjugate, exp, log)
        poly = updated
    if any(coefficient not in (0, 1) for coefficient in poly):
        raise RuntimeError("minimal polynomial did not land in GF(2)")
    return poly


def poly_mul_binary(left, right):
    result = [0] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        if not left_value:
            continue
        for j, right_value in enumerate(right):
            result[i + j] ^= left_value & right_value
    return result


def poly_to_int(poly):
    value = 0
    for degree, coefficient in enumerate(poly):
        if coefficient:
            value |= 1 << degree
    return value


def build_generator(exp, log):
    generator = [1]
    seen = set()
    for root_power in range(1, 11):
        coset = tuple(sorted((root_power * (2 ** shift)) % 31 for shift in range(5)))
        if coset in seen:
            continue
        seen.add(coset)
        generator = poly_mul_binary(
            generator, minimal_polynomial(exp[root_power], exp, log)
        )
    return generator


def gf2_remainder(value, generator):
    generator_degree = generator.bit_length() - 1
    while value.bit_length() - 1 >= generator_degree:
        shift = value.bit_length() - 1 - generator_degree
        value ^= generator << shift
    return value


def enumerate_code(generator, message_bits, code_length):
    generator_degree = generator.bit_length() - 1
    messages = np.arange(1 << message_bits, dtype=np.uint32)
    codewords = []
    for message in messages:
        shifted = int(message) << generator_degree
        codewords.append(shifted ^ gf2_remainder(shifted, generator))
    codes = np.asarray(codewords, dtype=np.uint32)
    if codes.max() >= 1 << code_length:
        raise RuntimeError("codeword exceeds code length")
    return codes


def bit_matrix(values, length):
    shifts = np.arange(length, dtype=np.uint32)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.int16)


def pairwise_distances(bits):
    weights = bits.sum(axis=1)
    return weights[:, None] + weights[None, :] - 2 * (bits @ bits.T)


def select_farthest(codes, count):
    bits = bit_matrix(codes, 30)
    distances = pairwise_distances(bits)
    weights = bits.sum(axis=1)
    selected = [int(np.argmax(weights))]
    minimum = distances[selected[0]].copy()
    while len(selected) < count:
        best_distance = int(minimum.max())
        candidates = np.flatnonzero(minimum == best_distance)
        chosen = int(candidates[0])
        selected.append(chosen)
        minimum = np.minimum(minimum, distances[chosen])
    return codes[np.asarray(selected, dtype=np.int64)], int(minimum.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=500)
    parser.add_argument("--message_length", type=int, default=30)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if args.message_length != 30:
        raise ValueError("this BCH construction targets a shortened 30-bit code")

    started = time.time()
    exp, log = build_gf32()
    generator_poly = build_generator(exp, log)
    generator = poly_to_int(generator_poly)
    message_bits = 31 - (generator.bit_length() - 1)
    if message_bits != 11:
        raise RuntimeError(
            f"expected BCH(31,11), got generator degree {generator.bit_length() - 1}"
        )

    codewords31 = enumerate_code(generator, message_bits, 31)
    shortened = ((codewords31 >> np.uint32(30)) & 1) == 0
    codewords30 = np.unique(
        codewords31[shortened] & np.uint32((1 << 30) - 1)
    )
    full_distances = pairwise_distances(bit_matrix(codewords30, 30))
    full_minimum = int(full_distances[np.triu_indices(codewords30.shape[0], k=1)].min())
    expected_size = 1 << (message_bits - 1)
    if codewords30.shape[0] != expected_size:
        raise RuntimeError(
            f"shortening produced {codewords30.shape[0]} codewords, "
            f"expected {expected_size}"
        )
    if full_minimum < 11:
        raise RuntimeError(f"BCH minimum distance is only {full_minimum}")

    selected, _ = select_farthest(codewords30, args.num_users)
    selected_bits = bit_matrix(selected, 30)
    selected_distances = pairwise_distances(selected_bits)
    upper = selected_distances[np.triu_indices(args.num_users, k=1)]
    minimum_distance = int(upper.min())
    if minimum_distance < 11:
        raise RuntimeError(f"selected set minimum distance is {minimum_distance}")

    output = {
        "method": "shortened_primitive_BCH_31_11_d11",
        "num_users": args.num_users,
        "message_length": 30,
        "full_codebook_size": int(codewords30.shape[0]),
        "full_codebook_min_distance": full_minimum,
        "generator_polynomial_hex": format(generator, "08x"),
        "min_distance": minimum_distance,
        "mean_distance": float(upper.mean()),
        "max_distance": int(upper.max()),
        "elapsed_seconds": time.time() - started,
        "codes": [
            format(int(value), "030b") for value in selected.astype(np.int64)
        ],
        "codes_hex": [
            format(int(value), "08x") for value in selected.astype(np.int64)
        ],
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(
        json.dumps(
            {
                key: output[key]
                for key in [
                    "num_users",
                    "full_codebook_size",
                    "full_codebook_min_distance",
                    "min_distance",
                    "mean_distance",
                    "max_distance",
                    "generator_polynomial_hex",
                    "elapsed_seconds",
                ]
            },
            indent=2,
        )
    )
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
