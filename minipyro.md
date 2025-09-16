# My notes on https://pyro.ai/examples/minipyro.html

## Line-by-line commentary of examples/minipyro.py

Full URL: https://github.com/pyro-ppl/pyro/blob/dev/examples/minipyro.py

**Line 16**: Note that `pyro.generic` redirects to `pyroapi`.
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

**Line 36**: By now, we've got `model` and `guide` defined, and we have `data`. `PYRO_STACK` and `PARAM_STORE` are empty.

**Line 51**: this is the first time that anything gets on the stack or the param-store.

Lets zoom into what happens.

### First call to SVI.step(data)

Looking in lines 302-319 of https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/minipyro.py

**Line 306**: Let's split it into three parts

 * The `trace()` constructor is called with no arguments.  That object has a `trace` attribute, an empty `OrderedDict` and is added to `PYRO_STACK`.  
 * The resulting `Messenger` is named `param_capture`.
 * Remember: at the conclusion of line 309, the `__exit__` method of `param_capture` will be called, removing it from the stack.  But it's still able to be referenced since it has a name.

**Line 308**: `block` is called which places a `block` `Messenger` object with `hide_fn` that blocks messages of type `"sample"`.  We won't be able to see this working until `apply_stack` gets called inside line 309.

Remember: at the conclusion of line 309, the `__exit__` method will be called, removing it from the stack.

