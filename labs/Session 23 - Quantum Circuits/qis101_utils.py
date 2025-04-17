"""qis101_utils.py"""

import numpy as np
from IPython.core.display import Math


def as_latex(a, places=5, column=False, prefix=""):
    def strip(val):
        # Handle common fractions
        if abs(np.round(val, 5)) == 0.25000:
            d = ""
            if val < 0:
                d += "-"
            d += r"\frac{1}{4}"
            return d
        if abs(np.round(val, 5)) == 0.50000:
            d = ""
            if val < 0:
                d += "-"
            d += r"\frac{1}{2}"
            return d
        if abs(np.round(val, 5)) == 0.57735:
            d = ""
            if val < 0:
                d += "-"
            d += r"\frac{1}{\sqrt{3}}"
            return d
        if abs(np.round(val, 5)) == 0.70711:
            d = ""
            if val < 0:
                d += "-"
            d += r"\frac{1}{\sqrt{2}}"
            return d
        if abs(np.round(val, 5)) == 0.8165:
            d = ""
            if val < 0:
                d += "-"
            d += r"\sqrt{\frac{2}{3}}"
            return d
        if abs(np.round(val, 5)) == 0.86603:
            d = ""
            if val < 0:
                d += "-"
            d += r"\frac{\sqrt{3}}{2}"
            return d
        frmt = ":." + str(places) + "f"
        d = str("{v" + frmt + "}").format(v=val)
        while d[-1] == "0":
            d = d[:-1]
        if d[-1] == ".":
            d = d[:-1]
        if float(d) == 0:
            d = "0"
        return d

    m = np.copy(a)
    if len(m.shape) == 1:
        m = m[np.newaxis, :]
        if column:
            m = m.T
    prec = 1 / 10**places
    s = r"\begin{bmatrix}"
    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            v = m[row, col]
            real_comp = float(np.real(v))
            imag_comp = float(np.imag(v))
            is_imag_neg = imag_comp < 0
            is_real_zero = bool(np.isclose(real_comp, 0, atol=prec))
            is_imag_zero = bool(np.isclose(imag_comp, 0, atol=prec))
            is_imag_one = bool(np.isclose(abs(imag_comp), 1, atol=prec))
            if is_real_zero:
                if is_imag_zero:
                    s += "0"
            else:
                s += strip(real_comp)
            if not is_imag_zero:
                if is_imag_one:
                    if is_imag_neg:
                        s += r"-i"
                    else:
                        if not is_real_zero:
                            s += "+"
                        s += r"i"
                else:
                    if not is_real_zero and not is_imag_neg:
                        s += " + "
                    s += strip(imag_comp) + "i"
            if col < m.shape[1] - 1:
                s += " &"
        if row < (m.shape[0] - 1):
            s += r"\\[1em]"
        else:
            s += r"\\"
    s += r"\end{bmatrix}"
    return Math(prefix + s)
