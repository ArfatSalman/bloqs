from random import uniform

import cirq
import numpy as np
import pytest
import qiskit.circuit.library as QiskitGates
from pyquil import Program
from pyquil.simulation.tools import program_unitary
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit_aer import Aer

from bloqs.common.QuantumGate import all_gates
from bloqs.ext.cirq import CirqGates
from bloqs.ext.pyquil import PyQuilGates, get_custom_get_definitions


def qiskit_unitary(gate_name, qubit_size, param_args, *, reversed_bits=True):
    qr = QuantumRegister(qubit_size, name="qr")
    cr = ClassicalRegister(qubit_size, name="cr")
    qc = QuantumCircuit(qr, cr)
    if reversed_bits:
        qc = qc.reverse_bits()
    if param_args:
        qc.append(
            getattr(QiskitGates, gate_name)(*param_args), qargs=list(qr), cargs=[]
        )
    else:
        qc.append(getattr(QiskitGates, gate_name)(), qargs=list(qr), cargs=[])

    qiskit_result = execute(qc, backend=Aer.get_backend("unitary_simulator")).result()

    return qiskit_result.get_unitary(qc).data


def cirq_unitary_using_bloqs(gate_name, qubit_size, param_args):
    q = cirq.LineQubit.range(qubit_size)
    qc = cirq.Circuit()
    if param_args:
        qc.append(getattr(CirqGates, gate_name)(*param_args)(*q))
    else:
        qc.append(getattr(CirqGates, gate_name)(*q))
    cirq_result = cirq.unitary(qc)
    return cirq_result


def pyquil_unitary_using_bloqs(gate_name, qubit_size, param_args):
    qc = Program()

    custom_gate_defn = get_custom_get_definitions(gate_name)
    if custom_gate_defn:
        qc += custom_gate_defn

    if param_args:
        if gate_name in [
            "RYYGate",
            "U2Gate",
            "U3Gate",
            "RXXGate",
            "UGate",
            "RGate",
        ]:
            qc += getattr(PyQuilGates, gate_name)(*param_args)(*range(qubit_size))
        else:
            qc += getattr(PyQuilGates, gate_name)(*param_args, *range(qubit_size))
    else:
        qc += getattr(PyQuilGates, gate_name)(*range(qubit_size))
    return program_unitary(qc, qubit_size)


gates = list(all_gates)


@pytest.mark.parametrize("gate_name", gates)
def test_cirq_gate(gate_name):
    fn_args = list(getattr(QiskitGates, gate_name).__init__.__code__.co_varnames)

    for name in ["self", "label", "ctrl_state", "basis", "OneQubitEulerDecomposer"]:
        if name in fn_args:
            fn_args.remove(name)

    params = len(fn_args)

    param_args = [uniform(-np.pi, np.pi) for _ in range(params)]
    if param_args:
        qubit_size = getattr(QiskitGates, gate_name)(*param_args).num_qubits
    else:
        qubit_size = getattr(QiskitGates, gate_name)().num_qubits

    unit_q = qiskit_unitary(gate_name, qubit_size, param_args)
    unit_c = cirq_unitary_using_bloqs(gate_name, qubit_size, param_args)
    assert np.allclose(unit_q, unit_c) or cirq.allclose_up_to_global_phase(
        unit_q, unit_c
    )


non_param_one_qubit_gates = ["SXGate", "SXdgGate"]

non_param_two_qubit_gates = ["CSXGate", "ECRGate"]

three_param_one_qubit_gates = [
    "UGate",
    "U3Gate",
]

two_param_one_qubit_gates = ["RGate", "U2Gate"]

tested_by_statevector = (
    two_param_one_qubit_gates
    + non_param_one_qubit_gates
    + non_param_two_qubit_gates
    + three_param_one_qubit_gates
)

pyquil_skipped_gates = tested_by_statevector + [
    "RZZGate",
    "RVGate",
    "C3SXGate",
    "RC3XGate",
    "CU3Gate",
    "RCCXGate",
    "RZXGate",
    "CUGate",
    "RYYGate",
    "RXXGate",
]

pyquil_gates = [gate for gate in gates if gate not in pyquil_skipped_gates]


@pytest.mark.parametrize("gate_name", pyquil_gates)
def test_pyquil_gates(gate_name):
    fn_args = list(getattr(QiskitGates, gate_name).__init__.__code__.co_varnames)

    for name in ["self", "label", "ctrl_state", "basis", "OneQubitEulerDecomposer"]:
        if name in fn_args:
            fn_args.remove(name)

    params = len(fn_args)

    param_args = [uniform(-np.pi, np.pi) for _ in range(params)]
    if param_args:
        qubit_size = getattr(QiskitGates, gate_name)(*param_args).num_qubits
    else:
        qubit_size = getattr(QiskitGates, gate_name)().num_qubits

    print(qubit_size)

    unit_q = qiskit_unitary(gate_name, qubit_size, param_args, reversed_bits=False)
    unit_pyq = pyquil_unitary_using_bloqs(gate_name, qubit_size, param_args)

    assert np.allclose(unit_q, unit_pyq) or cirq.allclose_up_to_global_phase(
        unit_q, unit_pyq
    )
