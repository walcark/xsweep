# Contract DSL: grammar and error catalogue

The string form is sugar over the `Contract` object. Both share the same
vocabulary, fixed during clarification (FR-006) and expensive to change once
the parser exists.

## Grammar

```ebnf
contract   ::= clause* "->" outputs
clause     ::= loop_cl | vec_cl | const_cl
loop_cl    ::= "loop" "(" name_list ")"
vec_cl     ::= "vec"  "(" vec_list  ")"
const_cl   ::= "const" "(" const_list ")"
outputs    ::= out_var ("," out_var)*
out_var    ::= name "(" [dim_list] ")"
vec_list   ::= vec_item ("," vec_item)*
vec_item   ::= name ["@" integer]
const_list ::= const_item ("," const_item)*
const_item ::= name ["(" [dim_list] ")"]
name_list  ::= name ("," name)*
dim_list   ::= name ("," name)*
name       ::= python_identifier
```

Tokens: `loop`, `vec`, `const`, `->`, `@`, `(`, `)`, `,`, identifiers,
integers. Whitespace is insignificant. Clause order is free; each clause
appears at most once.

## Reading the syntax

```
loop(aot, rh, sza) vec(wl @ 8) const(srf) -> tdir_down(wl)
```

- Clause parentheses contain VARIABLE names. Output parentheses contain DIM
  names. This asymmetry is deliberate and must stay documented: sweep dims
  are a runtime property of the space, never named in clauses.
- `-> out()` with empty parentheses means a scalar output per call.
- `vec(wl)` without `@ N` means the whole axis in one call.
- Multi-dim vec variables are allowed; their batch sizes come from the
  policy, never from `@ N` (FR-010).
- A const variable is handed whole, except on any dim it shares with a
  batched vec dim, where it auto-aligns to the active batch instead of
  causing a shape mismatch in the callable. `const(srf, bias(x, y))` marks
  `x` and `y` as protected on `bias`: those dims never slice, whatever the
  policy batches, which is the escape hatch for a const array that needs a
  full axis internally (a normalisation, a reduction) even while some other
  variable sharing that axis name is being batched.
- Statics are absent from the contract by design: they are configuration,
  not data.

## Vocabulary rationale

`vec` rather than `pass` because `pass` is a Python keyword, and rather than
`batch` because `batch` names the mechanism while `vec` names the physical
property that justifies it (the callee accepts a vector along this axis).
`const` rather than `whole` for the same reason: `whole` is the granularity,
`const` is the intent (context data). Recorded in design doc section 11.

## Error catalogue

Parse errors carry the character offset and the expected token. Coherence
errors carry the offending names.

| Situation | Message shape |
|-----------|---------------|
| Unknown clause keyword | `unknown clause 'batch' at offset 0; expected one of loop, vec, const` |
| Missing output clause | `contract has no '->' clause; every contract must name its outputs` |
| Duplicate clause | `clause 'vec' appears twice` |
| Name in two clauses | `variable 'wl' appears in both vec and const` |
| Duplicate output name | `output 'rho' declared twice` |
| `@ N` on a multi-dim vec variable | `'@ 8' on multi-dim variable 'A'; declare batch sizes per dim in the policy, e.g. chunks={"x": 500}` |
| `@ N` with `N < 1` | `batch size must be >= 1, got 0` |
| Batch on a reduced dim (1-D `@ N`) | `dim 'wl' is batched but absent from every declared output; the function reduces over it, so batching would corrupt the result` |
| Batch on a reduced dim (multi-dim `policy.chunks`) | `chunks batches dim 'y', which is absent from every declared output ['band']; the function reduces over it, so batching would corrupt the result` |
| Protected dim the const variable does not carry | `const 'bias' protects dim(s) ['z'], which it does not carry in the space; its dims are ['x', 'y']` |
| No outputs | `contract declares no output` |
| Invalid identifier | `'2wl' is not a valid name at offset 12` |

## Validation timing

Grammar and coherence checks run at parse time, which is import time for
modules (`__init_subclass__`) and decoration time for functions (FR-007). No
contract error may first appear during a sweep.

Checks that need the space (a vec variable's rank, a chunked dim's
existence, a const variable's protected dims, a reduced dim batched through
`policy.chunks`) run during planning, still before any call of the wrapped
function.
