from abc import ABC, abstractmethod
from collections import namedtuple

all_gates = {
    "C3XGate",
    "C3SXGate",
    "C4XGate",
    "CCXGate",
    "DCXGate",
    "CHGate",
    "CPhaseGate",
    "CRXGate",
    "CRYGate",
    "CRZGate",
    "CSwapGate",
    "CSXGate",
    "CUGate",
    "CU1Gate",
    "CU3Gate",
    "CXGate",
    "CYGate",
    "CZGate",
    "HGate",
    "IGate",
    # "MSGate",
    "PhaseGate",
    "RCCXGate",
    "RC3XGate",
    "RGate",
    "RVGate",
    "RXGate",
    "RXXGate",
    "RYGate",
    "RYYGate",
    "RZGate",
    "RZZGate",
    "RZXGate",
    # "XXPlusYYGate",
    # "XXMinusYYGate",
    "ECRGate",
    "SGate",
    "CSGate",
    "SdgGate",
    "CSdgGate",
    "SwapGate",
    "iSwapGate",
    "SXGate",
    "SXdgGate",
    "TGate",
    "TdgGate",
    "UGate",
    "U1Gate",
    "U2Gate",
    "U3Gate",
    "XGate",
    "YGate",
    "ZGate",
}


def QuantumGateFactory():
    attrs = {}
    for gate in all_gates:

        def meth():
            raise NotImplementedError()

        attrs[gate] = staticmethod(abstractmethod(meth))
    return type("", (ABC,), attrs)


QuantumGate = QuantumGateFactory()
