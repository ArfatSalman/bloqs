import math
from cmath import exp

import cirq
import numpy as np
from cirq.circuits.qasm_output import QasmUGate

from bloqs.common.QuantumGate import QuantumGate


def gate_from_impl(clazz):
    def inner(*args):
        return clazz()(*args)

    return inner


def CRXGate(angle):
    return cirq.ControlledGate(cirq.rx(angle), num_controls=1)


def CRYGate(angle):
    return cirq.ControlledGate(cirq.ry(angle), num_controls=1)


def CRZGate(angle):
    return cirq.ControlledGate(cirq.rz(angle), num_controls=1)


class PhaseGate(cirq.Gate):
    def __init__(self, lam):
        super()
        self.lam = lam

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        """Return a numpy.array for the Phase gate."""
        lam = float(self.lam)
        return np.array([[1, 0], [0, exp(1j * lam)]])

    def _resolve_parameters_(self, param_resolver, recursive):
        return PhaseGate(
            cirq.resolve_parameters(self.lam, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return f"Phase({self.lam:.2f})"

    def __pow__(self, power):
        if power == -1:
            return PhaseGate(-self.lam)
        return super().__pow__(power)


class CPhaseGate(cirq.Gate):
    def __init__(self, eith):
        super()
        self.eith = eith

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        a, b = qubits

        yield PhaseGate(self.eith / 2)(a)
        yield cirq.CX(a, b)
        yield PhaseGate(-self.eith / 2)(b)
        yield cirq.CX(a, b)
        yield PhaseGate(self.eith / 2)(b)

    def _resolve_parameters_(self, param_resolver, recursive):
        return CPhaseGate(
            cirq.resolve_parameters(self.eith, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return "@", f"CPhase({self.eith:.2f})"


class U1Gate(cirq.Gate):
    def __init__(self, lam):
        super()
        self.lam = lam

    def _num_qubits_(self):
        return 1

    def _decompose_(self, qubits):
        (a,) = qubits
        yield PhaseGate(self.lam)(a)

    def _resolve_parameters_(self, param_resolver, recursive):
        return U1Gate(
            cirq.resolve_parameters(self.lam, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return f"U1({self.lam:.2f})"


class U2Gate(cirq.Gate):
    def __init__(self, phi, lam):
        super()
        self.phi = phi
        self.lam = lam

    def _num_qubits_(self):
        return 1

    def _decompose_(self, qubits):
        (a,) = qubits
        yield QasmUGate(1 / 2, self.phi / np.pi, self.lam / np.pi)(a)

    def _circuit_diagram_info_(self, args):
        return f"U2({self.lam:.2f})"


class CUGate(cirq.Gate):
    def __init__(self, theta, phi, lam, gamma):
        super()
        self.theta = theta
        self.phi = phi
        self.lam = lam
        self.gamma = gamma

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        theta, phi, lam, gamma = self.theta, self.phi, self.lam, self.gamma
        c, t = qubits
        yield PhaseGate(gamma)(c)
        yield PhaseGate((lam + phi) / 2)(c)
        yield PhaseGate((lam - phi) / 2)(t)
        yield cirq.CX(c, t)
        yield QasmUGate((-theta / 2) / np.pi, 0, (-(phi + lam) / 2) / np.pi)(t)
        yield cirq.CX(c, t)
        yield QasmUGate((theta / 2) / np.pi, phi / np.pi, 0)(t)

    def _resolve_parameters_(self, param_resolver, recursive):
        return CUGate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.phi, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.lam, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.gamma, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return "@", f"CU"


class UGate(cirq.Gate):
    def __init__(self, theta, phi, lam):
        super()
        self.theta = theta
        self.phi = phi
        self.lam = lam

    def _num_qubits_(self):
        return 1

    def _decompose_(self, qubits):
        (a,) = qubits
        yield QasmUGate(self.theta / np.pi, self.phi / np.pi, self.lam / np.pi)(a)

    # def _unitary_(self):
    #     """Return a numpy.array for the U gate."""
    #     theta, phi, lam = self.theta, self.phi, self.lam
    #     cos = math.cos(theta / 2)
    #     sin = math.sin(theta / 2)
    #     return np.array(
    #         [
    #             [cos, -exp(1j * lam) * sin],
    #             [exp(1j * phi) * sin, exp(1j * (phi + lam)) * cos],
    #         ],
    #         dtype=complex,
    #     )

    def _resolve_parameters_(self, param_resolver, recursive):
        return UGate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.phi, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.lam, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return "U"


class CU3Gate(cirq.Gate):
    def __init__(self, theta, phi, lam):
        super()
        self.theta = theta
        self.phi = phi
        self.lam = lam

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        p_theta, p_phi, p_lambda = self.theta, self.phi, self.lam
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [
                    0,
                    0,
                    np.cos(p_theta / 2),
                    -np.exp(1j * p_lambda) * np.sin(p_theta / 2),
                ],
                [
                    0,
                    0,
                    np.exp(1j * p_phi) * np.sin(p_theta / 2),
                    np.exp(1j * p_lambda + 1j * p_phi) * np.cos(p_theta / 2),
                ],
            ]
        )

    def _resolve_parameters_(self, param_resolver, recursive):
        return CU3Gate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.phi, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.lam, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return "C3U"


class DCXGateImpl(cirq.Gate):
    def __init__(self):
        super()

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.CX(a, b)
        yield cirq.CX(b, a)

    def _circuit_diagram_info_(self, args):
        return "@", "DCX"


DCXGate = gate_from_impl(DCXGateImpl)


class RGate(cirq.Gate):
    def __init__(self, theta, phi):
        super()
        self.theta = theta
        self.phi = phi

    def _num_qubits_(self):
        return 1

    def _decompose_(self, qubits):
        (a,) = qubits
        theta = self.theta
        phi = self.phi
        yield UGate(theta, phi - np.pi / 2, -phi + np.pi / 2)(a)

    def _resolve_parameters_(self, param_resolver, recursive):
        return RGate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.phi, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return f"R({self.theta})"


class RVGate(cirq.Gate):
    def __init__(
        self,
        v_x,
        v_y,
        v_z,
    ):
        super()
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        v = np.asarray((self.v_x, self.v_y, self.v_z), dtype=float)
        angle = np.sqrt(v.dot(v))
        if angle == 0:
            return np.array([[1, 0], [0, 1]])
        nx, ny, nz = v / angle
        sin = np.sin(angle / 2)
        cos = np.cos(angle / 2)
        return np.array(
            [
                [cos - 1j * nz * sin, (-ny - 1j * nx) * sin],
                [(ny - 1j * nx) * sin, cos + 1j * nz * sin],
            ]
        )

    def _resolve_parameters_(self, param_resolver, recursive):
        return RVGate(
            cirq.resolve_parameters(self.v_x, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.v_y, param_resolver, recursive=recursive),
            cirq.resolve_parameters(self.v_z, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return f"RV"


class RZXGate(cirq.Gate):
    def __init__(self, theta):
        super()
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        # gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(1) q1; cx q0,q1; h q1; }
        q0, q1 = qubits

        yield cirq.H(q1)
        yield cirq.CX(q0, q1)
        yield cirq.rz(self.theta)(q1)
        yield cirq.CX(q0, q1)
        yield cirq.H(q1)

    def _resolve_parameters_(self, param_resolver, recursive):
        return RZXGate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return "@", "RZX"


class RCCXGateImpl(cirq.Gate):
    def __init__(self):
        super()

    def _num_qubits_(self):
        return 3

    def _decompose_(self, qubits):
        pi = np.pi
        a, b, c = qubits

        yield QasmUGate(1 / 2, 0 / np.pi, pi / np.pi)(c)
        yield U1Gate(pi / 4)(c)
        yield cirq.CX(b, c)
        yield U1Gate(-pi / 4)(c)
        yield cirq.CX(a, c)
        yield U1Gate(pi / 4)(c)
        yield cirq.CX(b, c)
        yield U1Gate(-pi / 4)(c)
        yield QasmUGate(1 / 2, 0 / np.pi, pi / np.pi)(c)

    def _circuit_diagram_info_(self, args):
        return "@", "@", "RC3X"


RCCXGate = gate_from_impl(RCCXGateImpl)


class RC3XGateImpl(cirq.Gate):
    def __init__(self):
        super()

    def _num_qubits_(self):
        return 4

    def _decompose_(self, qubits):
        pi = np.pi
        a, b, c, d = qubits
        yield QasmUGate(1 / 2, 0 / np.pi, pi / np.pi)(d)
        yield U1Gate(pi / 4)(d)
        yield cirq.CX(c, d)
        yield U1Gate(-pi / 4)(d)
        yield QasmUGate(1 / 2, 0 / np.pi, pi / np.pi)(d)
        yield cirq.CX(a, d)
        yield U1Gate(pi / 4)(d)
        yield cirq.CX(b, d)
        yield U1Gate(-pi / 4)(d)
        yield cirq.CX(a, d)
        yield U1Gate(pi / 4)(d)
        yield cirq.CX(b, d)
        yield U1Gate(-pi / 4)(d)
        yield QasmUGate(1 / 2, 0 / np.pi, pi / np.pi)(d)
        yield U1Gate(pi / 4)(d)
        yield cirq.CX(c, d)
        yield U1Gate(-pi / 4)(d)
        yield QasmUGate(1 / 2, 0 / np.pi, pi / np.pi)(d)

    def _circuit_diagram_info_(self, args):
        return "@", "@", "@", "RC3X"


RC3XGate = gate_from_impl(RC3XGateImpl)


class RZZGate(cirq.Gate):
    def __init__(self, theta):
        super()
        self.theta = theta

    def _num_qubits_(self):
        return 2

    # def _decompose_(self, qubits):
    #     a, b = qubits
    #     yield cirq.CX(a, b)
    #     u1(theta) b; cx a, b; }

    def _unitary_(self):
        """Return a numpy.array for the RZZ gate."""

        itheta2 = 1j * float(self.theta) / 2
        return np.array(
            [
                [exp(-itheta2), 0, 0, 0],
                [0, exp(itheta2), 0, 0],
                [0, 0, exp(itheta2), 0],
                [0, 0, 0, exp(-itheta2)],
            ]
        )

    def _resolve_parameters_(self, param_resolver, recursive):
        return RZZGate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
        )

    def __pow__(self, power):
        if power == -1:
            return RZZGate(-self.theta)
        return super().__pow__(power)

    def _circuit_diagram_info_(self, args):
        return "@", f"RZZ({self.theta:.2f})"


class ECRGateImpl(cirq.Gate):
    def __init__(self):
        super()

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        pi = np.pi
        q0, q1 = qubits

        yield RZXGate(pi / 4)(q0, q1)
        yield cirq.X(q0)
        yield RZXGate(-pi / 4)(q0, q1)

    def _circuit_diagram_info_(self, args):
        return "@", "ECR"


ECRGate = gate_from_impl(ECRGateImpl)


class RYYGate(cirq.Gate):
    def __init__(self, theta):
        super(RYYGate, self)
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        """Return a numpy.array for the RYY gate."""
        theta = float(self.theta)
        cos = math.cos(theta / 2)
        isin = 1j * math.sin(theta / 2)
        return np.array(
            [
                [cos, 0, 0, isin],
                [0, cos, -isin, 0],
                [0, -isin, cos, 0],
                [isin, 0, 0, cos],
            ],
        )

    def _circuit_diagram_info_(self, args):
        return "@", "RYY"

    def _resolve_parameters_(self, param_resolver, recursive):
        return RYYGate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
        )

    def __pow__(self, power):
        if power == -1:
            return RYYGate(-self.theta)
        return super().__pow__(power)


class CU1Gate(cirq.Gate):
    def __init__(self, eith):
        super()
        self.eith = eith

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        a, b = qubits
        angle = self.eith

        yield U1Gate(angle / 2)(a)
        yield cirq.CX(a, b)
        yield U1Gate(-angle / 2)(b)
        yield cirq.CX(a, b)
        yield U1Gate(angle / 2)(b)

    def _resolve_parameters_(self, param_resolver, recursive):
        return CU1Gate(
            cirq.resolve_parameters(self.eith, param_resolver, recursive=recursive),
        )

    def _circuit_diagram_info_(self, args):
        return f"CU1({self.eith:.2f})", "CU1"


class RXXGate(cirq.Gate):
    def __init__(self, theta):
        super(RXXGate, self)
        self.theta = theta

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        """Return a Numpy.array for the RXX gate."""
        theta2 = float(self.theta) / 2
        cos = math.cos(theta2)
        isin = 1j * math.sin(theta2)
        return np.array(
            [
                [cos, 0, 0, -isin],
                [0, cos, -isin, 0],
                [0, -isin, cos, 0],
                [-isin, 0, 0, cos],
            ]
        )

    def _resolve_parameters_(self, param_resolver, recursive):
        return RXXGate(
            cirq.resolve_parameters(self.theta, param_resolver, recursive=recursive),
        )

    def __pow__(self, power):
        if power == -1:
            return RXXGate(-self.theta)
        return super().__pow__(power)

    def _circuit_diagram_info_(self, args):
        return "@", f"RXX({self.theta:.2f})"


implementations = {
    # Natively Available Gates
    "IGate": cirq.I,
    "TGate": cirq.T,
    "TdgGate": cirq.T ** (-1),
    "SGate": cirq.S,
    "CSGate": cirq.ControlledGate(cirq.S, num_controls=1),
    "SdgGate": cirq.S ** (-1),
    "HGate": cirq.H,
    "CHGate": cirq.ControlledGate(cirq.H, num_controls=1),
    "XGate": cirq.X,
    "CXGate": cirq.CX,
    "CCXGate": cirq.CCX,
    "CSdgGate": cirq.ControlledGate(cirq.S ** (-1), num_controls=1),
    "SXdgGate": cirq.X ** (-1 / 2),
    "SXGate": cirq.X**0.5,
    "CSXGate": cirq.CX**0.5,
    "C3XGate": cirq.X.controlled(num_controls=3),
    "C3SXGate": cirq.X.controlled(num_controls=3) ** 0.5,
    "C4XGate": cirq.X.controlled(num_controls=4),
    "YGate": cirq.Y,
    "CYGate": cirq.ControlledGate(cirq.Y, num_controls=1),
    "ZGate": cirq.Z,
    "CZGate": cirq.CZ,
    "RXGate": cirq.rx,
    "RYGate": cirq.ry,
    "RZGate": cirq.rz,
    "SwapGate": cirq.SWAP,
    "iSwapGate": cirq.ISWAP,
    "CSwapGate": cirq.ControlledGate(cirq.SWAP, num_controls=1),
    "CRXGate": CRXGate,
    "CRYGate": CRYGate,
    "CRZGate": CRZGate,
    "RGate": RGate,
    "RVGate": RVGate,
    "RXXGate": RXXGate,
    "RYYGate": RYYGate,
    "RZZGate": RZZGate,
    "CU1Gate": CU1Gate,
    "ECRGate": ECRGate,
    "RC3XGate": RC3XGate,
    "RCCXGate": RCCXGate,
    "RZXGate": RZXGate,
    "DCXGate": DCXGate,
    "U3Gate": UGate,
    "CU3Gate": CU3Gate,
    "UGate": UGate,
    "CUGate": CUGate,
    "U2Gate": U2Gate,
    "U1Gate": U1Gate,
    "CPhaseGate": CPhaseGate,
    "PhaseGate": PhaseGate,
}


def create_class():
    attrs = {}
    for k, v in implementations.items():
        attrs[k] = staticmethod(v)
    return type("CirqGates", (QuantumGate,), attrs)


CirqGates = create_class()
