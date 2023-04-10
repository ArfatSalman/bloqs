from random import uniform

import cirq
import numpy as np
import pytest
import qiskit.circuit.library as QiskitGates
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit_aer import Aer

from bloqs.common.QuantumGate import all_gates
from bloqs.ext.cirq import CirqGates
from bloqs.ext.pyquil import PyQuilGates


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


gates = list(all_gates)

@pytest.mark.parametrize("gate_name", gates)
def test_gate(gate_name):
    fn_args = list(getattr(QiskitGates, gate_name).__init__.__code__.co_varnames)

    for name in ["self", "label", "ctrl_state", "basis", 'OneQubitEulerDecomposer']:

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
