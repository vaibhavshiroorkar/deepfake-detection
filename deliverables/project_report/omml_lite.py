"""A small LaTeX-subset to Office Math (OMML) converter.

It covers what the report needs: letters, digits, Greek letters, operators, sub- and superscripts, fractions, roots,
n-ary operators (sum, integral) with limits and an operand, delimiters (\\left ... \\right), accents (hat, bar, tilde),
\\underbrace, \\mathrm, \\mathbf, \\mathbb, \\mathcal, \\text and spacing commands. The output is a single m:oMath element
string (no namespace declarations; the caller declares xmlns:m).
"""
import re
from xml.sax.saxutils import escape

GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "varepsilon": "ε", "epsilon": "ε", "theta": "θ",
         "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "omega": "ω",
         "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "ell": "ℓ", "ln": None}
SYM = {"le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "approx": "≈", "to": "→", "longrightarrow": "⟶", "infty": "∞",
       "in": "∈", "top": "⊤", "succeq": "⪰", "cdot": "·", "times": "×", "sim": "∼", "setminus": "∖", "mid": "|",
       "dots": "…", "ldots": "…", "cdots": "⋯", "partial": "∂", "nabla": "∇", "pm": "±", "ne": "≠", "neq": "≠",
       "#": "#", "%": "%", "{": "{", "}": "}", "|": "|", "star": "∗", "ll": "≪", "gg": "≫", "forall": "∀", "cup": "∪"}
FUNCS = {"ln", "max", "min", "exp", "log", "sin", "cos", "Pr", "arg"}
BB = {"E": "\U0001d53c", "R": "ℝ", "P": "ℙ", "N": "ℕ"}
CAL = {"T": "\U0001d4af", "B": "ℬ", "D": "\U0001d49f", "I": "ℐ", "L": "ℒ", "N": "\U0001d4a9", "S": "\U0001d4ae", "F": "ℱ"}
STOP = {"=", "+", "−", ",", "≈", "≤", "≥", "→", "⟶", "∈", "⪰", "≠"}
SPACE = {"quad": " ", "qquad": "  ", ";": " ", ",": " ", " ": " ", ":": " "}


def run(text, sty=None, nor=False):
    pr = ""
    if nor:
        pr = "<m:rPr><m:nor/></m:rPr>"
    elif sty:
        pr = '<m:rPr><m:sty m:val="%s"/></m:rPr>' % sty
    return '<m:r>%s<m:t xml:space="preserve">%s</m:t></m:r>' % (pr, escape(text))


class Atom:
    def __init__(self, xml, text="", kind="ord"):
        self.xml, self.text, self.kind = xml, text, kind
        self.nary = None  # dict for n-ary operators


class Parser:
    def __init__(self, s):
        self.s = s
        self.i = 0

    # -- helpers
    def peek(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\n":
            self.i += 1
        return self.s[self.i] if self.i < len(self.s) else ""

    def raw_group(self):
        """Raw text of a braced group (nesting respected); the opening brace is next."""
        assert self.peek() == "{", self.s[self.i:self.i + 20]
        depth, j = 0, self.i
        while True:
            if self.s[j] == "{":
                depth += 1
            elif self.s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out = self.s[self.i + 1:j]
        self.i = j + 1
        return out

    def arg(self):
        """One argument: a braced group or a single atom, as xml."""
        if self.peek() == "{":
            self.i += 1
            atoms = self.seq(until="}")
            return "".join(a.xml for a in atoms)
        a = self.atom()
        return a.xml if a else ""

    def arg_atoms(self):
        if self.peek() == "{":
            self.i += 1
            return self.seq(until="}")
        a = self.atom()
        return [a] if a else []

    # -- parsing
    def seq(self, until=None):
        atoms = []
        while True:
            c = self.peek()
            if c == "":
                break
            if until == "}" and c == "}":
                self.i += 1
                break
            if until == "right" and self.s.startswith("\\right", self.i):
                break
            a = self.atom()
            if a is None:
                continue
            a = self.scripts(a)
            atoms.append(a)
        self.absorb(atoms)
        return atoms

    def scripts(self, a):
        sub = sup = None
        while self.peek() in ("_", "^"):
            c = self.s[self.i]
            self.i += 1
            part = self.arg()
            if c == "_":
                sub = part
            else:
                sup = part
        if sub is None and sup is None:
            return a
        if a.nary is not None:
            a.nary["sub"], a.nary["sup"] = sub, sup
            return a
        base = a.xml
        if a.kind == "func" and sub is not None and sup is None:
            return Atom('<m:limLow><m:e>%s</m:e><m:lim>%s</m:lim></m:limLow>' % (base, sub), a.text, "ord")
        if sub is not None and sup is not None:
            x = '<m:sSubSup><m:e>%s</m:e><m:sub>%s</m:sub><m:sup>%s</m:sup></m:sSubSup>' % (base, sub, sup)
        elif sub is not None:
            x = '<m:sSub><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>' % (base, sub)
        else:
            x = '<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>' % (base, sup)
        return Atom(x, a.text, "ord")

    def absorb(self, atoms):
        """Gives every n-ary operator its operand: the following atoms up to a relation, a plus or a minus."""
        for k in range(len(atoms) - 1, -1, -1):
            a = atoms[k]
            if a.nary is not None and a.nary.get("e") is None:
                j = k + 1
                depth = 0
                while j < len(atoms):
                    t = atoms[j]
                    if t.kind == "ord" and t.text in ("(", "["):
                        depth += 1
                    elif t.kind == "ord" and t.text in (")", "]"):
                        depth -= 1
                    elif depth <= 0 and ((t.kind == "op" and t.text in STOP) or (t.kind == "space" and t.text in (" ", "  "))):
                        break
                    j += 1
                operand = atoms[k + 1:j]
                del atoms[k + 1:j]
                n = a.nary
                a.xml = ('<m:nary><m:naryPr><m:chr m:val="%s"/><m:limLoc m:val="%s"/><m:grow m:val="1"/>%s%s</m:naryPr><m:sub>%s</m:sub><m:sup>%s</m:sup><m:e>%s</m:e></m:nary>'
                         % (n["chr"], n["loc"], "<m:subHide m:val=\"1\"/>" if not n.get("sub") else "", "<m:supHide m:val=\"1\"/>" if not n.get("sup") else "",
                            n.get("sub") or "", n.get("sup") or "", "".join(o.xml for o in operand)))
                n["e"] = True

    def atom(self):
        c = self.peek()
        if c == "":
            return None
        if c == "{":
            self.i += 1
            atoms = self.seq(until="}")
            return Atom("".join(a.xml for a in atoms), "", "group")
        if c == "\\":
            return self.command()
        self.i += 1
        if c.isdigit():
            j = self.i
            while j < len(self.s) and (self.s[j].isdigit() or (self.s[j] == "." and j + 1 < len(self.s) and self.s[j + 1].isdigit())):
                j += 1
            t = self.s[self.i - 1:j]
            self.i = j
            return Atom(run(t), t, "ord")
        if c.isalpha():
            if c.isupper() and self.i < len(self.s) and self.s[self.i].isupper():   # TP, FN, ...: an abbreviation, upright
                j = self.i
                while j < len(self.s) and self.s[j].isupper():
                    j += 1
                t = self.s[self.i - 1:j]
                self.i = j
                return Atom(run(t, "p"), t, "ord")
            return Atom(run(c), c, "ord")
        if c == "-":
            return Atom(run("−"), "−", "op")
        if c == "*":
            return Atom(run("∗"), "*", "ord")
        if c == "'":
            return Atom(run("′"), "'", "ord")
        if c in "=+,:<>":
            return Atom(run(c), c, "op")
        return Atom(run(c), c, "ord")

    def command(self):
        self.i += 1  # backslash
        m = re.match(r"[A-Za-z]+", self.s[self.i:])
        if not m:
            ch = self.s[self.i]
            self.i += 1
            if ch in SPACE:
                return Atom(run(SPACE[ch], nor=True), SPACE[ch], "space")
            if ch == "!":
                return None
            return Atom(run(SYM.get(ch, ch)), ch, "ord")
        name = m.group(0)
        self.i += len(name)
        if name in SPACE:
            return Atom(run(SPACE[name], nor=True), SPACE[name], "space")
        if name in GREEK and GREEK[name]:
            return Atom(run(GREEK[name]), GREEK[name], "ord")
        if name in SYM:
            t = SYM[name]
            return Atom(run(t), t, "op" if t in STOP else "ord")
        if name in ("mathrm", "operatorname", "mathit", "mathbf", "text", "textbf"):
            raw = self.raw_group()
            raw = raw.replace("\\,", " ").replace("\\ ", " ").replace("\\;", " ")
            if name == "text":
                return Atom(run(raw, nor=True), raw, "ord")
            sty = {"mathrm": "p", "operatorname": "p", "mathit": "i", "mathbf": "b", "textbf": "b"}[name]
            return Atom(run(raw, sty), raw, "ord")
        if name == "mathbb":
            raw = self.raw_group().strip()
            return Atom(run(BB.get(raw, raw), "p"), raw, "ord")
        if name == "mathcal":
            raw = self.raw_group().strip()
            return Atom(run(CAL.get(raw, raw), "p"), raw, "ord")
        if name in FUNCS:
            if name == "arg" and self.s.startswith("\\min", self.i):
                self.i += 4
                return Atom(run("arg min", "p"), "argmin", "func")
            if name in ("max", "min", "arg"):
                return Atom(run(name, "p"), name, "func")
            return Atom(run(" ", nor=True) + run(name, "p") + run(" ", nor=True), name, "ord")
        if name in ("frac", "tfrac", "dfrac"):
            n, d = self.arg(), self.arg()
            return Atom('<m:f><m:fPr><m:type m:val="bar"/></m:fPr><m:num>%s</m:num><m:den>%s</m:den></m:f>' % (n, d), "", "ord")
        if name == "sqrt":
            return Atom('<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>%s</m:e></m:rad>' % self.arg(), "", "ord")
        if name in ("hat", "widehat", "tilde", "bar", "overline"):
            e = self.arg()
            if name in ("bar", "overline"):
                return Atom('<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>%s</m:e></m:bar>' % e, "", "ord")
            chr_ = "̂" if name in ("hat", "widehat") else "̃"
            return Atom('<m:acc><m:accPr><m:chr m:val="%s"/></m:accPr><m:e>%s</m:e></m:acc>' % (chr_, e), "", "ord")
        if name == "underbrace":
            e = self.arg()
            x = ('<m:groupChr><m:groupChrPr><m:chr m:val="⏟"/><m:pos m:val="bot"/><m:vertJc m:val="top"/></m:groupChrPr><m:e>%s</m:e></m:groupChr>' % e)
            if self.peek() == "_":
                self.i += 1
                lab = self.arg()
                x = '<m:limLow><m:e>%s</m:e><m:lim>%s</m:lim></m:limLow>' % (x, lab)
            return Atom(x, "", "ord")
        if name in ("sum", "int", "prod"):
            ch = {"sum": "∑", "int": "∫", "prod": "∏"}[name]
            a = Atom("", name, "ord")
            a.nary = {"chr": ch, "loc": "undOvr" if name != "int" else "subSup"}
            return a
        if name == "left":
            d = self.peek()
            if self.s.startswith("\\", self.i):
                self.i += 1
                d = self.s[self.i]
            self.i += 1
            beg = "" if d == "." else d
            atoms = self.seq(until="right")
            self.i += len("\\right")
            e = self.peek()
            if self.s[self.i] == "\\":
                self.i += 1
                e = self.s[self.i]
            else:
                e = self.s[self.i]
            self.i += 1
            end = "" if e == "." else e
            x = '<m:d><m:dPr><m:begChr m:val="%s"/><m:endChr m:val="%s"/></m:dPr><m:e>%s</m:e></m:d>' % (escape(beg), escape(end), "".join(a.xml for a in atoms))
            return Atom(x, "", "ord")
        raise ValueError("unsupported command \\%s in %r" % (name, self.s))


def omml(latex):
    p = Parser(latex)
    atoms = p.seq()
    return "<m:oMath>%s</m:oMath>" % "".join(a.xml for a in atoms)
