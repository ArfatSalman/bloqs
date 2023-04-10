from functools import reduce


def get_qiskit_like_output(results_list):
    results = list(
        map(
            lambda arr: reduce(lambda x, y: str(x) + str(y), arr[::-1], ""),
            results_list,
        )
    )
    counts = dict(zip(results, [results.count(i) for i in results]))
    return counts
