from random import uniform

import numpy as np
import pytest

import qiskit.circuit.library as QiskitGates
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer

from pyquil import Program
from pyquil.gates import X
from pyquil.api import WavefunctionSimulator

from qiskit import QuantumCircuit, QuantumRegister, execute

from bloqs.ext.pyquil import PyQuilGates, get_custom_get_definitions

one_qubit_gates = []


class OneQubitGateTest:
    def __init__(self):
        self.basis_states = [[1, 0], [0, 1]]  # |0>  # |1>

    def qiskit_statevector(self, gate, initial_state):
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        qc.initialize(initial_state, qr[0])
        qc.append(gate, qargs=[qr[0]])
        return Statevector.from_instruction(qc).data

    def pyquil_statevector(self, gate, gate_name, init_state):
        p = Program()
        p += get_custom_get_definitions(gate_name)

        # manually create the initial state
        if init_state == [0, 1]:  # |1>
            p += X(0)

        p += gate
        wfs = WavefunctionSimulator()
        return wfs.wavefunction(p).amplitudes

    def compare(self, gate_name, params=None):
        tolerance = 1e-9

        for state in self.basis_states:
            if params:
                qiskit_result = self.qiskit_statevector(
                    getattr(QiskitGates, gate_name)(*params), state
                )
                pyquil_result = self.pyquil_statevector(
                    getattr(PyQuilGates, gate_name)(*params)(0), gate_name, state
                )
            else:
                qiskit_result = self.qiskit_statevector(
                    getattr(QiskitGates, gate_name)(), state
                )
                pyquil_result = self.pyquil_statevector(
                    getattr(PyQuilGates, gate_name)(0), gate_name, state
                )
            # print(pyquil_result)
            dot_product = np.dot(np.conj(qiskit_result), pyquil_result)
            # print(dot_product)
            overlap = np.abs(dot_product)

            if not np.isclose(overlap, 1, atol=tolerance):
                return False

        return True


from .test_bloqs import (
    non_param_one_qubit_gates,
    three_param_one_qubit_gates,
    two_param_one_qubit_gates,
)


@pytest.mark.parametrize("gate_name", non_param_one_qubit_gates)
def test_non_param_one_qubit_gates(gate_name):
    tester = OneQubitGateTest()

    assert tester.compare(gate_name)


@pytest.mark.parametrize(
    "gate_name", two_param_one_qubit_gates + three_param_one_qubit_gates
)
def test_three_param_one_qubit_gates(gate_name):
    tester = OneQubitGateTest()
    if gate_name in three_param_one_qubit_gates:
        p = 3
    else:
        p = 2

    assert tester.compare(gate_name, [uniform(-np.pi, np.pi) for _ in range(p)])
