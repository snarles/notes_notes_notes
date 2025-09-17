# My notes on https://pyro.ai/examples/minipyro.html

## Stepping through execution

Source Files: 

| Path | URL | Abbreviation |
|-|-|-|
| examples/minipyro.py | https://github.com/pyro-ppl/pyro/blob/dev/examples/minipyro.py | E |
| pyro/contrib/minipyro.py | https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/minipyro.py | M |

**E-16**: Note that `pyro.generic` redirects to `pyroapi`.
Looking at
https://github.com/pyro-ppl/pyro-api/blob/master/pyroapi/dispatch.py, we have the following definitions

```
register_backend('minipyro', {
    'distributions': 'pyro.distributions',
    'handlers': 'pyro.poutine',
    'infer': 'pyro.contrib.minipyro',
    'ops': 'torch',
    'optim': 'pyro.contrib.minipyro',
    'pyro': 'pyro.contrib.minipyro',
})
```

**E-36**: By now, we've got `model` and `guide` defined, and we have `data`. `PYRO_STACK` and `PARAM_STORE` are empty.

**E-51**: this is the first time that anything gets on the stack or the param-store.

**M-306**: `SVI.step(data)`

 * The `trace()` constructor is called with no arguments.  That object has a `trace` attribute, an empty `OrderedDict` and is added to `PYRO_STACK`.  
 * The resulting `Messenger` is named `param_capture`.

**M-308**: `block` is called which places a `block` `Messenger` object with `hide_fn` that blocks messages of type `"sample"`, let's call it `foo_block`.  We won't be able to see this working until `apply_stack` gets called inside line 309.


**M-331**: `elbo(model, guide, data)`

 * A `trace` object is constructed with `fn=guide`.  For our purposes, name it as `foo_gt`.
 * A call to `get_trace(data)` causes `foo_gt` to pass `data` as an argument to `__call__`.
 * Inside `__call__(data)`, M-63 `with self:` causes the `__enter__` method of `foo_gt` to be called.  At this point, then `PYRO_STACK==[param_capture, foo_block, foo_gt]`.
 * Inside `__call__(data)`, M-64, calls `guide(data)` to be returned as `guide_trace`.
 * Inside `guide(data)`, M-32 of examples/minipyro,py, we have a call to `sample("loc", dist.Normal(guide_loc, guide_scale))`.
 * Inside `sample("loc", dist.Normal(guide_loc, guide_scale))`, M-187: nothing happens, as there is no `obs`.
 * Inside `sample("loc", dist.Normal(guide_loc, guide_scale))`, M-204: there is a call to `apply_stack(initial_msg)`.  See *M-204*.  `initial_msg` in M-204 gets a random value from `dist.Normal(guide_loc, guide_scale)()`, and `foo_gt.trace["loc"]` gets set to `initial_msg`.
 * Done with `apply_stack(initial_msg)` and the result is called `msg`.
 * Done with `sample("loc", dist.Normal(guide_loc, guide_scale))`, return `msg["value"]` but it's not stored.
 * Done with `guide(data)`, no return.
 * Done with `with self:` the `__exit__` method of `foo_gt` is called.
 * Done with `__call__(data)`, returns `None` since `guide` has no return.
 * Done with `get_trace(data)`, now `guide_trace` is `foo_gt.trace`.
 * Right now, `PYRO_STACK`

Note that `guide_trace` has the value
```
{
    "loc":
        {
            "type": "sample",
            "name": "loc",
            "fn": dist.Normal(guide_loc, guide_scale),
            "args": None,
            "kwargs": None,
            "value": [some random value],
            "stop": True
        }   
}
```

**M-204**: `apply_stack(initial_msg)`

Note that `initial_msg` has the values
```
    {
        "type": "sample",
        "name": "loc",
        "fn": dist.Normal(guide_loc, guide_scale),
        "args": None,
        "kwargs": None,
        "value": None,
    }
```

 * M-167: `(pointer, handler)` are going to be `(0, foo_gt)`, `(1, foo_block)`, `(2, param_capture)`.
 * Loop: `pointer`=`0`, `handler`=`foo_gt`.  pass
 * Loop: `pointer`=`1`, `handler`=`foo_block`.  Since `initial_msg["type"]` is `"sample"`, `foo_block.hide_fn(initial_msg)` evaluates to `True`.  Hence, `initial_msg["stop"]` gets set to `True`.
 * M-171: Since `initial_msg["stop"]` is `True`, we break out of the `for` loop.
 * M-174: `dist.Normal(guide_loc, guide_scale)()` is called, sampling a normal random variate and storing that to `initial_msg["value"]`.
 * M-179: the for statement evaluates to `for handler in PYRO_STACK[-2:]:` and hence it will loop over `foo_block` and `foo_gt`.
 * Loop: `handler`=`foo_block`: pass
 * Loop: `handler`=`foo_gt`: set `foo_gt.trace["loc"]` to `initial_msg`
 * Returns `initial_msg`.


**M-309**: 

 * `foo_block.__exit__()` method called, removing `foo_block` from the stack.
 * `param_capture.__exit__()` called, removing `param_capture` from the stack.  But it's still able to be referenced since it has a name.

**E-57**
