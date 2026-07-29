"""Contract objects and the string DSL that builds them.

A contract describes ONE CALL: how each input variable is consumed, and what
the call produces. It never describes the data layout, which xarray dims
already encode.

The tokeniser, the parser and the coherence checks live here together on
purpose: they produce one error catalogue, and splitting them would scatter
those messages across two modules.

Grammar (see ``specs/001-xsweep-v0/contracts/contract-dsl.md``)::

    loop(aot, rh, sza) vec(wl @ 8) const(srf) -> tdir_down(wl)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .errors import ContractError

__all__ = ["Contract", "LoopVar", "OutVar", "VecVar"]

_CLAUSES = ("loop", "vec", "const")
_TOKEN_RE = re.compile(
    r"""
    (?P<arrow>->)
  | (?P<at>@)
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<comma>,)
  | (?P<int>\d+)
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<space>\s+)
  | (?P<bad>.)
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    """One lexical unit, with the offset needed to point at it in errors."""

    kind: str
    text: str
    offset: int


@dataclass(frozen=True)
class LoopVar:
    """A variable consumed one value per call.

    Parameters
    ----------
    name
        Variable name in the space.
    deliver
        ``"scalar"`` hands the callable a native Python scalar, which is what
        external engines want; ``"array"`` keeps a 0-d DataArray.
    """

    name: str
    deliver: Literal["scalar", "array"] = "scalar"


@dataclass(frozen=True)
class VecVar:
    """A variable whose axes the callable accepts as a vector.

    Parameters
    ----------
    name
        Variable name in the space.
    max_batch
        Default batch size along the variable's single dim. ``None`` means the
        whole axis in one call. Only meaningful for 1-D variables; multi-dim
        vec variables take their sizes from the policy instead.
    """

    name: str
    max_batch: int | None = None


@dataclass(frozen=True)
class OutVar:
    """A named output with its call-level dims.

    Parameters
    ----------
    name
        Name the produced variable takes in the result.
    dims
        Dims one call produces, excluding the loop dims that the sweeper
        prepends itself.
    sizes
        Declared sizes, aligned with ``dims``. ``None`` entries are discovered
        by a probe call at execution time.
    """

    name: str
    dims: tuple[str, ...]
    sizes: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class Contract:
    """Frozen description of one call, including its physics revision.

    Parameters
    ----------
    loop
        Variables consumed one value per call.
    vec
        Variables consumed as vectors, possibly in batches.
    const
        Variables handed whole to every call.
    out
        Named outputs with their call-level dims.
    version
        User-declared revision of the computation. Bump it whenever the
        result of the wrapped callable changes for reasons xsweep cannot
        observe: the function body, an external engine binary, a data file.
        It enters the fingerprint, so bumping it invalidates the cache.
    """

    loop: tuple[LoopVar, ...] = ()
    vec: tuple[VecVar, ...] = ()
    const: tuple[str, ...] = ()
    out: tuple[OutVar, ...] = ()
    version: str = "0"

    @property
    def inputs(self) -> tuple[str, ...]:
        """Return every input variable name, in clause order."""
        return (
            tuple(v.name for v in self.loop)
            + tuple(v.name for v in self.vec)
            + self.const
        )

    @property
    def outputs(self) -> tuple[str, ...]:
        """Return every declared output name."""
        return tuple(o.name for o in self.out)

    @property
    def out_dims(self) -> tuple[str, ...]:
        """Return the union of the declared output dims, in order of first use."""
        seen: dict[str, None] = {}
        for o in self.out:
            for d in o.dims:
                seen.setdefault(d, None)
        return tuple(seen)

    @classmethod
    def parse(cls, spec: str, *, version: str = "0") -> Contract:
        """Build a contract from its string form.

        Parameters
        ----------
        spec
            Contract string, e.g. ``"loop(aot, rh) vec(wl @ 8) -> t(wl)"``.
        version
            Physics revision, see the class docstring.

        Returns
        -------
        Contract
            The parsed and validated contract.

        Raises
        ------
        ContractError
            If the string cannot be parsed or is incoherent.
        """
        contract = _Parser(spec).parse(version=version)
        _check_coherence(contract)
        return contract

    def render(self) -> str:
        """Return the canonical string form, used in reports and fingerprints."""
        parts: list[str] = []
        if self.loop:
            parts.append(f"loop({', '.join(v.name for v in self.loop)})")
        if self.vec:
            items = [
                v.name if v.max_batch is None else f"{v.name} @ {v.max_batch}"
                for v in self.vec
            ]
            parts.append(f"vec({', '.join(items)})")
        if self.const:
            parts.append(f"const({', '.join(self.const)})")
        outs = ", ".join(f"{o.name}({', '.join(o.dims)})" for o in self.out)
        return f"{' '.join(parts)} -> {outs}".strip()


def coerce(spec: str | Contract, *, version: str = "0") -> Contract:
    """Accept either form of a contract and return the object.

    Parameters
    ----------
    spec
        A contract string or an already-built contract.
    version
        Physics revision, applied only when parsing a string. Passing a
        non-default version alongside a ``Contract`` that carries its own is
        rejected, since silently picking one would hide a real conflict.

    Returns
    -------
    Contract
        The contract object.

    Raises
    ------
    ContractError
        If both the object and the argument declare a version and they differ.
    """
    if isinstance(spec, Contract):
        if version != "0" and spec.version != version:
            raise ContractError(
                f"contract already declares version {spec.version!r} but "
                f"version={version!r} was passed; declare it in one place only"
            )
        return spec
    return Contract.parse(spec, version=version)


def _tokenise(spec: str) -> list[Token]:
    """Split a contract string into tokens, rejecting stray characters."""
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(spec):
        kind = m.lastgroup
        assert kind is not None
        if kind == "space":
            continue
        if kind == "bad":
            raise ContractError(
                f"unexpected character {m.group()!r} at offset {m.start()}"
            )
        tokens.append(Token(kind, m.group(), m.start()))
    return tokens


@dataclass
class _Parser:
    """Recursive-descent parser over the token stream."""

    spec: str
    pos: int = 0
    tokens: list[Token] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Tokenise the input eagerly so offsets are available everywhere."""
        self.tokens = _tokenise(self.spec)

    def parse(self, *, version: str) -> Contract:
        """Parse the whole contract string."""
        clauses: dict[str, list[object]] = {}
        while self._peek() is not None and self._peek_kind() != "arrow":
            name_tok = self._expect("name", "a clause keyword")
            if name_tok.text not in _CLAUSES:
                raise ContractError(
                    f"unknown clause {name_tok.text!r} at offset "
                    f"{name_tok.offset}; expected one of {', '.join(_CLAUSES)}"
                )
            if name_tok.text in clauses:
                raise ContractError(f"clause {name_tok.text!r} appears twice")
            self._expect("lparen", "'(' after the clause keyword")
            clauses[name_tok.text] = self._clause_body(name_tok.text)
            self._expect("rparen", "')' closing the clause")

        if self._peek() is None:
            raise ContractError(
                "contract has no '->' clause; every contract must name its outputs"
            )
        self._expect("arrow", "'->'")
        out = self._outputs()

        loop = tuple(LoopVar(str(n)) for n in clauses.get("loop", ()))
        vec = tuple(v for v in clauses.get("vec", ()) if isinstance(v, VecVar))
        const = tuple(str(n) for n in clauses.get("const", ()))
        return Contract(loop=loop, vec=vec, const=const, out=out, version=version)

    def _clause_body(self, clause: str) -> list[object]:
        """Parse the comma-separated names inside one clause."""
        items: list[object] = []
        while True:
            tok = self._expect("name", "a variable name")
            if clause == "vec":
                max_batch: int | None = None
                if self._peek_kind() == "at":
                    self._next()
                    n_tok = self._expect("int", "a batch size after '@'")
                    max_batch = int(n_tok.text)
                    if max_batch < 1:
                        raise ContractError(f"batch size must be >= 1, got {max_batch}")
                items.append(VecVar(tok.text, max_batch))
            else:
                items.append(tok.text)
            if self._peek_kind() != "comma":
                return items
            self._next()

    def _outputs(self) -> tuple[OutVar, ...]:
        """Parse the output clause: one or more ``name(dims)`` entries."""
        outs: list[OutVar] = []
        while True:
            name = self._expect("name", "an output name")
            self._expect("lparen", "'(' after the output name")
            dims: list[str] = []
            if self._peek_kind() != "rparen":
                while True:
                    dims.append(self._expect("name", "a dim name").text)
                    if self._peek_kind() != "comma":
                        break
                    self._next()
            self._expect("rparen", "')' closing the output dims")
            outs.append(OutVar(name.text, tuple(dims), (None,) * len(dims)))
            if self._peek_kind() != "comma":
                break
            self._next()
        if self._peek() is not None:
            tok = self._peek()
            assert tok is not None
            raise ContractError(
                f"unexpected {tok.text!r} at offset {tok.offset} after the "
                "output clause"
            )
        return tuple(outs)

    def _peek(self) -> Token | None:
        """Return the next token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _peek_kind(self) -> str | None:
        """Return the kind of the next token, or ``None`` at end of input."""
        tok = self._peek()
        return tok.kind if tok is not None else None

    def _next(self) -> Token:
        """Consume and return the next token."""
        tok = self._peek()
        if tok is None:
            raise ContractError("unexpected end of contract")
        self.pos += 1
        return tok

    def _expect(self, kind: str, what: str) -> Token:
        """Consume the next token, requiring it to be of ``kind``."""
        tok = self._peek()
        if tok is None:
            raise ContractError(f"unexpected end of contract; expected {what}")
        if tok.kind != kind:
            raise ContractError(
                f"expected {what} at offset {tok.offset}, got {tok.text!r}"
            )
        self.pos += 1
        return tok


def _check_coherence(contract: Contract) -> None:
    """Validate a contract against the rules that need no space.

    Parameters
    ----------
    contract
        The freshly parsed contract.

    Raises
    ------
    ContractError
        If a name appears in two clauses, an output is declared twice, there
        is no output, or a batched dim is absent from every output.
    """
    seen: dict[str, str] = {}
    for clause, names in (
        ("loop", [v.name for v in contract.loop]),
        ("vec", [v.name for v in contract.vec]),
        ("const", list(contract.const)),
    ):
        for name in names:
            if name in seen:
                raise ContractError(
                    f"variable {name!r} appears in both {seen[name]} and {clause}"
                )
            seen[name] = clause

    if not contract.out:
        raise ContractError("contract declares no output")

    out_seen: set[str] = set()
    for o in contract.out:
        if o.name in out_seen:
            raise ContractError(f"output {o.name!r} declared twice")
        out_seen.add(o.name)

    # '@ N' is the 1-D shorthand, so the name it carries must be both a
    # variable and an output dim. Without a space we cannot tell which of the
    # two readings failed, so the message names both: either the callable
    # reduces over that axis, in which case batching would silently corrupt
    # the result (edge case 12), or the variable is multi-dim, in which case
    # the marker cannot say which dim it meant (FR-010).
    out_dims = set(contract.out_dims)
    for v in contract.vec:
        if v.max_batch is not None and v.name not in out_dims:
            raise ContractError(
                f"'@ {v.max_batch}' on {v.name!r}, which is not among the "
                f"declared output dims {sorted(out_dims)!r}. Either the "
                "function reduces over that axis, and batching it would "
                "corrupt the result, so drop the marker to pass the whole "
                f"axis in one call; or {v.name!r} is multi-dim, and batch "
                "sizes must then be declared per dim in the policy, e.g. "
                'chunks={"x": 500, "y": 500}'
            )
