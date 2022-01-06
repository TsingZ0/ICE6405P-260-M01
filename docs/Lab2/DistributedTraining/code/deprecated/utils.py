from typing import Callable, Any
import torch.distributed.rpc as rpc

# --------- Helper Methods --------------------


# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods.
def call_method(method: Callable, rref: rpc.RRef, *args, **kwargs) -> Any:
    return method(rref.local_value(), *args, **kwargs)


# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef. args and kwargs are passed into the method.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.


def remote_method(method: Callable, rref: rpc.RRef, *args, **kwargs) -> Any:
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)
