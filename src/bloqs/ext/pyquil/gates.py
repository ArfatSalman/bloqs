from pyquil import Program
from pyquil.gates import (
    I,
    X,
    Y,
    Z,
    H,
    S,
    T,
    RX,
    RY,
    RZ,
    PHASE,
    CZ,
    CNOT,
    CCNOT,
    CPHASE,
    SWAP,
    CSWAP,
    ISWAP,
)
from pyquil.quil import DefGate
from pyquil.quilatom import Parameter, quil_sin, quil_cos, quil_exp
from bloqs.common.QuantumGate import QuantumGate
import numpy


def not_implemented(*args, **kwargs):
    raise NotImplementedError()


def C3X(a, b, c, d):
    return CCNOT(b, c, d).controlled(a)


def C4X(a, b, c, d, e):
    return CCNOT(c, d, e).controlled(b).controlled(a)


def CH(a, b):
    return H(b).controlled(a)


def CRX(angle, a, b):
    return RX(angle, b).controlled(a)


def CRY(angle, a, b):
    return RY(angle, b).controlled(a)


def CRZ(angle, a, b):
    return RZ(angle, b).controlled(a)


def CY(a, b):
    return Y(b).controlled(a)

def CS(a, b):
    return S(b).controlled(a)


def ECR():
    part_mat = numpy.array(
        [[0, 0, 1, 1.0j], [0, 0, 1.0j, 1], [1, -1.0j, 0, 0], [-1.0j, 1, 0, 0]],
    )
    mat = 1 / numpy.sqrt(2) * part_mat
    return DefGate("ECR", mat)


def U(theta=Parameter("theta"), phi=Parameter("phi"), lam=Parameter("lam"), name="U"):

    cos = quil_cos(theta / 2)
    sin = quil_sin(theta / 2)
    u_mat = numpy.array(
        [
            [cos, -quil_exp(1j * lam) * sin],
            [quil_exp(1j * phi) * sin, quil_exp(1j * (phi + lam)) * cos],
        ]
    )
    args = [el for el in [theta, phi, lam] if isinstance(el, Parameter)]
    return DefGate(name, u_mat, args)


def SX():
    mat = numpy.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
    sqrt_x_definition = DefGate("SX", mat)
    return sqrt_x_definition


def DCX(a, b):
    p = Program()
    p += CNOT(a, b)
    p += CNOT(b, a)
    return p


def Sdg(a):
    return S(a).dagger()


def CSdg(a, b):
    return Sdg(b).controlled(a)


def Tdg(a):
    return T(a).dagger()

def CCZ(a, b, c):
    return Z(a).controlled(b).controlled(c)

def RXX():
    theta = Parameter("theta")
    theta2 = theta / 2
    cos = quil_cos(theta2)
    isin = 1j * quil_sin(theta2)
    mat = numpy.array(
        [
            [cos, 0, 0, -isin],
            [0, cos, -isin, 0],
            [0, -isin, cos, 0],
            [-isin, 0, 0, cos],
        ],
    )
    return DefGate("RXX", mat, [theta])


def RYY():
    theta = Parameter("theta")
    cos = quil_cos(theta / 2)
    isin = 1j * quil_sin(theta / 2)
    mat = numpy.array(
        [[cos, 0, 0, isin], [0, cos, -isin, 0], [0, -isin, cos, 0], [isin, 0, 0, cos]],
    )
    return DefGate("RYY", mat, [theta])


def RZZ():
    theta = Parameter("theta")
    itheta2 = 1j * theta / 2
    mat = numpy.array(
        [
            [quil_exp(-itheta2), 0, 0, 0],
            [0, quil_exp(itheta2), 0, 0],
            [0, 0, quil_exp(itheta2), 0],
            [0, 0, 0, quil_exp(-itheta2)],
        ],
    )
    return DefGate("RZZ", mat, [theta])


def RZX():
    theta = Parameter("theta")
    half_theta = theta / 2
    cos = quil_cos(half_theta)
    isin = 1j * quil_sin(half_theta)
    mat = numpy.array(
        [[cos, -isin, 0, 0], [-isin, cos, 0, 0], [0, 0, cos, isin], [0, 0, isin, cos]],
    )
    return DefGate("RZX", mat, [theta])


def RCCX():
    mat = numpy.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1j],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1j, 0, 0, 0, 0],
        ],
    )
    return DefGate("RCCX", mat)


def RC3X():
    mat = numpy.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    return DefGate("RC3X", mat)


def RV(v_x, v_y, v_z):
    def inner(v_x, v_y, v_z):
        raise NotImplementedError("RV is not yet implemented")

    return inner


def R(theta, phi):
    U3Gate = defns["UGate"].get_constructor()
    return U3Gate(theta, phi - numpy.pi / 2, -phi + numpy.pi / 2)


defns = {
    "UGate": U(),
    "SXGate": SX(),
    "CSXGate": SX(),
    "ECRGate": ECR(),
    "U2Gate": U(theta=numpy.pi / 2, name="U2"),
    "RXXGate": RXX(),
    "RYYGate": RYY(),
    "RZZGate": RZZ(),
    "RZXGate": RZX(),
    "RCCXGate": RCCX(),
    "RC3XGate": RC3X(),
}


def CU3(theta, phi, lam, c, t):
    return (implementations["UGate"](theta, phi, lam)(t)).controlled(c)


def CU(theta, phi, lam, gamma, c, t):
    p = Program()

    p += PHASE(gamma, c)
    p += PHASE((lam + phi) / 2, c)
    p += PHASE((lam - phi) / 2, t)
    p += CNOT(c, t)
    p += implementations["UGate"](-theta / 2, 0, -(phi + lam) / 2)(t)
    p += CNOT(c, t)
    p += implementations["UGate"](theta / 2, phi, 0)(t)
    return p


def SXdgGate(a):
    return defns["SXGate"].get_constructor()(a).dagger()


def CSX(a, b):
    sx_gate = defns["SXGate"].get_constructor()
    return sx_gate(b).controlled(a)


def C3SX(a, b, c, d):
    sx_gate = defns["SXGate"].get_constructor()
    return sx_gate(d).controlled([c, b, a])


def get_custom_get_definitions(*args):
    res = []
    args = set(args)

    r_gate = {"RGate"}
    if len(set.intersection(r_gate, args)) > 0:
        res.append(defns["UGate"])

        args.difference_update(r_gate)

    sx_group = {"SXGate", "CSXGate", "SXdgGate", "C3SXGate"}
    if len(set.intersection(sx_group, args)) > 0:
        res.append(defns["SXGate"])

        args.difference_update(sx_group)

    u_group = {"UGate", "U3Gate", "CUGate", "CU3Gate"}
    if len(set.intersection(u_group, args)) > 0:
        res.append(defns["UGate"])
        args.difference_update(u_group)

    for gate in args:
        if defns.get(gate):
            res.append(defns.get(gate))

    return res


implementations = {
    "C3SXGate": C3SX,
    "C3XGate": C3X,
    "C4XGate": C4X,
    "CCXGate": CCNOT,
    "CCZGate": CCZ,
    "CHGate": CH,
    "CPhaseGate": CPHASE,
    "CRXGate": CRX,
    "CRYGate": CRY,
    "CRZGate": CRZ,
    "CSXGate": CSX,
    "CSwapGate": CSWAP,
    "CU1Gate": CPHASE,
    "CUGate": CU,
    "CU3Gate": CU3,
    "CXGate": CNOT,
    "CYGate": CY,
    "CZGate": CZ,
    "DCXGate": DCX,
    "ECRGate": defns["ECRGate"].get_constructor(),
    "HGate": H,
    "IGate": I,
    "PhaseGate": PHASE,
    "RC3XGate": defns["RC3XGate"].get_constructor(),
    "RCCXGate": defns["RCCXGate"].get_constructor(),
    "RVGate": RV,
    "RGate": R,
    "RXGate": RX,
    "RXXGate": defns["RXXGate"].get_constructor(),
    "RYGate": RY,
    "RYYGate": defns["RYYGate"].get_constructor(),
    "RZGate": RZ,
    "RZXGate": defns["RZXGate"].get_constructor(),
    "RZZGate": defns["RZZGate"].get_constructor(),
    "SGate": S,
    "CSGate": CS,
    "SXGate": defns["SXGate"].get_constructor(),
    "SXdgGate": SXdgGate,
    "SdgGate": Sdg,
    "CSdgGate": CSdg,
    "SwapGate": SWAP,
    "TGate": T,
    "TdgGate": Tdg,
    "U1Gate": PHASE,
    "U2Gate": defns["U2Gate"].get_constructor(),
    "U3Gate": defns["UGate"].get_constructor(),
    "UGate": defns["UGate"].get_constructor(),
    "XGate": X,
    "YGate": Y,
    "ZGate": Z,
    "iSwapGate": ISWAP,
}


def create_class():
    attrs = {}
    for k, v in implementations.items():
        attrs[k] = staticmethod(v)
    return type("PyQuilGates", (QuantumGate,), attrs)


PyQuilGates = create_class()
