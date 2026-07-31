# API reference

Everything below is exported from the `xsweep` top-level namespace. Nothing
else is public.

## Declaring a sweep

```{eval-rst}
.. autofunction:: xsweep.sweep
```

```{eval-rst}
.. autoclass:: xsweep.Sweeper
   :members: explain, __call__
```

```{eval-rst}
.. autoclass:: xsweep.SweepModule
   :members: explain, forward, __call__
```

## Configuring a run

```{eval-rst}
.. autoclass:: xsweep.SweepPolicy
   :members:
```

## Inspecting a run before it happens

```{eval-rst}
.. autoclass:: xsweep.Plan
   :members:
```

## The parsed contract

```{eval-rst}
.. autoclass:: xsweep.Contract
   :members:
```

```{eval-rst}
.. autoclass:: xsweep.LoopVar
.. autoclass:: xsweep.VecVar
.. autoclass:: xsweep.OutVar
```

## Errors

Every error raised by xsweep derives from `XsweepError`, so a caller can
catch the whole family with one `except`.

```{eval-rst}
.. autoexception:: xsweep.XsweepError
.. autoexception:: xsweep.ContractError
.. autoexception:: xsweep.PolicyError
.. autoexception:: xsweep.SpaceError
.. autoexception:: xsweep.StoreError
.. autoexception:: xsweep.StoreLockedError
.. autoexception:: xsweep.PointFailed
```
