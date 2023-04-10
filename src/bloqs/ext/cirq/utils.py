from functools import reduce


def get_qiskit_like_output(result, keys):
    result_dict = dict(
        result.multi_measurement_histogram(keys=keys)
    )

    keys = list(
        map(lambda arr: reduce(lambda x, y: str(x) +
            str(y), arr[::-1]), result_dict.keys())
    )
    return dict(zip( map(str, keys), result_dict.values()))
