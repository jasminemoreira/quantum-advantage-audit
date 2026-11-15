"""Exact sparse engine over Z[zeta_16] (13 Nov 2026).

Same scheme as the integer-width note (amplitude u_x/sqrt(2)^k, u_x made of 8 integers, global
k, reduction by parity), but it stores only the nonzero amplitudes in a dictionary
index -> tuple of 8 Python integers (arbitrary precision). It is meant for circuits that are large
in qubits and small in support, such as the RQGA ones (23 to 27 qubits, most of them ancillas in |0>).

It includes the exact product in the ring, conjugation and exact marginal probabilities:
P(y) = (sum of u_x * conj(u_x) over x with projection y) / 2^k, returned as (numerator in
Z[zeta], k). The probability is real, so the numerator lies in the real subring; to test
equality with a rational r = a/2^b it suffices to compare 2^b * numerator with a * 2^k * (1,0,...,0).
"""
from __future__ import annotations


def zeta_m(u, m):
    """u (tuple of 8 ints) times zeta^m."""
    m %= 16
    neg = m >= 8
    m %= 8
    v = tuple(-u[8 - m + j] for j in range(m)) + tuple(u[:8 - m]) if m else tuple(u)
    return tuple(-a for a in v) if neg else v


def mult(u, v):
    """Product in Z[zeta_16] = Z[x]/(x^8+1)."""
    r = [0] * 8
    for i, a in enumerate(u):
        if a:
            for j, b in enumerate(v):
                if b:
                    s = i + j
                    if s < 8:
                        r[s] += a * b
                    else:
                        r[s - 8] -= a * b
    return tuple(r)


def conj(u):
    """conj(zeta^j) = zeta^(16-j) = -zeta^(8-j) for j>=1."""
    return (u[0],) + tuple(-u[8 - j] for j in range(1, 8))


class MotorEsparso:
    def __init__(self, n):
        self.n = n
        self.s = {0: (1, 0, 0, 0, 0, 0, 0, 0)}
        self.k = 0
        self.max_suporte = 1

    def h(self, q):
        b = 1 << q
        novo = {}
        for x in {x & ~b for x in self.s}:
            u0 = self.s.get(x, (0,) * 8)
            u1 = self.s.get(x | b, (0,) * 8)
            a = tuple(p + r for p, r in zip(u0, u1))
            d = tuple(p - r for p, r in zip(u0, u1))
            if any(a):
                novo[x] = a
            if any(d):
                novo[x | b] = d
        self.s = novo
        self.k += 1
        self.max_suporte = max(self.max_suporte, len(novo))

    def fase(self, m, qs):
        mask = 0
        for q in qs:
            mask |= 1 << q
        for x, u in self.s.items():
            if x & mask == mask:
                self.s[x] = zeta_m(u, m)

    def mcx(self, cs, t):
        mask = 0
        for c in cs:
            mask |= 1 << c
        bt = 1 << t
        self.s = {(x ^ bt if x & mask == mask else x): u for x, u in self.s.items()}

    def reduz(self):
        while self.k > 0:
            if any((u[j] - u[j + 4]) & 1 for u in self.s.values() for j in range(4)):
                break
            novo = {}
            for x, u in self.s.items():
                w2, w6 = zeta_m(u, 2), zeta_m(u, 6)
                novo[x] = tuple((a >> 1) - (b >> 1) for a, b in zip(w2, w6))
            self.s = novo
            self.k -= 1

    def aplica(self, p):
        if p[0] == "h":
            self.h(p[1])
        elif p[0] == "fase":
            self.fase(p[1], p[2])
        else:
            self.mcx(p[1], p[2])
        self.reduz()

    def soma_quadrados(self):
        return sum(a * a for u in self.s.values() for a in u)

    def marginal(self, qubits):
        """Exact numerators {y: sum u conj(u)} in the projection onto `qubits` (in the given order)."""
        out = {}
        for x, u in self.s.items():
            y = 0
            for i, q in enumerate(qubits):
                y |= ((x >> q) & 1) << i
            nu = mult(u, conj(u))
            acc = out.get(y, (0,) * 8)
            out[y] = tuple(a + b for a, b in zip(acc, nu))
        return out

    def max_bits(self):
        return max((abs(a) for u in self.s.values() for a in u), default=0).bit_length()
