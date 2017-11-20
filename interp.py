#!/usr/bin/env python3
# Author: Rahul Gopinath <rahul.gopinath@cispa.saarland>
# License: GPLv3
"""
Meta Circular Interpreter
"""

__g__ = """
-- ASDL's 7 builtin types are:
-- identifier, int, string, bytes, object, singleton, constant
--
-- singleton: None, True or False
-- constant can be None, whereas None means "no value" for object.

module Python
{
    mod = Module(stmt* body)
        | Interactive(stmt* body)
        | Expression(expr body)

        -- not really an actual node but useful in Jython's typesystem.
        | Suite(stmt* body)

    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns)
          | AsyncFunctionDef(identifier name, arguments args,
                             stmt* body, expr* decorator_list, expr? returns)

          | ClassDef(identifier name,
             expr* bases,
             keyword* keywords,
             stmt* body,
             expr* decorator_list)
          | Return(expr? value)

          | Delete(expr* targets)
          | Assign(expr* targets, expr value)
          | AugAssign(expr target, operator op, expr value)
          -- 'simple' indicates that we annotate simple name without parens
          | AnnAssign(expr target, expr annotation, expr? value, int simple)

          -- use 'orelse' because else is a keyword in target languages
          | For(expr target, expr iter, stmt* body, stmt* orelse)
          | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse)
          | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
          | With(withitem* items, stmt* body)
          | AsyncWith(withitem* items, stmt* body)

          | Raise(expr? exc, expr? cause)
          | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | Assert(expr test, expr? msg)

          | Import(alias* names)
          | ImportFrom(identifier? module, alias* names, int? level)

          | Global(identifier* names)
          | Nonlocal(identifier* names)
          | Expr(expr value)
          | Pass | Break | Continue

          -- XXX Jython will be different
          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset)

          -- BoolOp() can use left & right?
    expr = BoolOp(boolop op, expr* values)
         | BinOp(expr left, operator op, expr right)
         | UnaryOp(unaryop op, expr operand)
         | Lambda(arguments args, expr body)
         | IfExp(expr test, expr body, expr orelse)
         | Dict(expr* keys, expr* values)
         | Set(expr* elts)
         | ListComp(expr elt, comprehension* generators)
         | SetComp(expr elt, comprehension* generators)
         | DictComp(expr key, expr value, comprehension* generators)
         | GeneratorExp(expr elt, comprehension* generators)
         -- the grammar constrains where yield expressions can occur
         | Await(expr value)
         | Yield(expr? value)
         | YieldFrom(expr value)
         -- need sequences for compare to distinguish between
         -- x < 4 < 3 and (x < 4) < 3
         | Compare(expr left, cmpop* ops, expr* comparators)
         | Call(expr func, expr* args, keyword* keywords)
         | Num(object n) -- a number as a PyObject.
         | Str(string s) -- need to specify raw, unicode, etc?
         | FormattedValue(expr value, int? conversion, expr? format_spec)
         | JoinedStr(expr* values)
         | Bytes(bytes s)
         | NameConstant(singleton value)
         | Ellipsis
         | Constant(constant value)

         -- the following expression can appear in assignment context
         | Attribute(expr value, identifier attr, expr_context ctx)
         | Subscript(expr value, slice slice, expr_context ctx)
         | Starred(expr value, expr_context ctx)
         | Name(identifier id, expr_context ctx)
         | List(expr* elts, expr_context ctx)
         | Tuple(expr* elts, expr_context ctx)

          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset)

    expr_context = Load | Store | Del | AugLoad | AugStore | Param

    slice = Slice(expr? lower, expr? upper, expr? step)
          | ExtSlice(slice* dims)
          | Index(expr value)

    boolop = And | Or

    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                 | RShift | BitOr | BitXor | BitAnd | FloorDiv

    unaryop = Invert | Not | UAdd | USub

    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn

    comprehension = (expr target, expr iter, expr* ifs, int is_async)

    excepthandler = ExceptHandler(expr? type, identifier? name, stmt* body)
                    attributes (int lineno, int col_offset)

    arguments = (arg* args, arg? vararg, arg* kwonlyargs, expr* kw_defaults,
                 arg? kwarg, expr* defaults)

    arg = (identifier arg, expr? annotation)
           attributes (int lineno, int col_offset)

    -- keyword arguments supplied to call (NULL identifier for **kwargs)
    keyword = (identifier? arg, expr value)

    -- import name with optional 'as' alias.
    alias = (identifier name, identifier? asname)

    withitem = (expr context_expr, expr? optional_vars)
}
"""

import ast
import sys
import operator

class Interpreter:
    """
    The meta circular python interpreter
    """
    def __init__(self):
        self.OpMap = {
          ast.Is: lambda a, b: a is b,
          ast.IsNot: lambda a, b: a is not b,
          ast.In: lambda a, b: a in b,
          ast.NotIn: lambda a, b: a not in b,
          ast.Add: lambda a, b: a + b,
          ast.BitAnd: lambda a, b: a & b,
          ast.BitOr: lambda a, b: a | b,
          ast.BitXor: lambda a, b: a ^ b,
          ast.Div: lambda a, b: a / b,
          ast.FloorDiv: lambda a, b: a // b,
          ast.LShift:  lambda a, b: a << b,
          ast.RShift: lambda a, b: a >> b,
          ast.Mult:  lambda a, b: a * b,
          ast.Pow: lambda a, b: a ** b,
          ast.Sub: lambda a, b: a - b,
          ast.Mod: lambda a, b: a % b,
          ast.And: lambda a, b: a and b,
          ast.Or: lambda a, b: a or b,
          ast.Eq: lambda a, b: a == b,
          ast.Gt: lambda a, b: a > b,
          ast.GtE: lambda a, b: a >= b,
          ast.Lt: lambda a, b: a < b,
          ast.LtE: lambda a, b: a <= b,
          ast.NotEq: lambda a, b: a != b,
          ast.Invert: lambda a: ~a,
          ast.Not: lambda a: not a,
          ast.UAdd: lambda a: +a,
          ast.USub: lambda a: -a}
        self.symtable = {
           'print': print
        }

    def parse(self, src):
        """
        >>> i = Interpreter()
        >>> v = i.parse('123')
        >>> v.body[0].value.n
        123
        """
        return ast.parse(src)

    def on_module(self, node):
        """
        Module(stmt* body)
        """
        # return value of module is the last statement
        return self.body_eval(node.body)

    def on_num(self, node):
        """
        Num(object n) -- a number as a PyObject.
        >>> i = Interpreter()
        >>> i.eval('123')
        123
        """
        return node.n

    def on_assign(self, node):
        """
        Assign(expr* targets, expr value)
        TODO: AugAssign(expr target, operator op, expr value)
        -- 'simple' indicates that we annotate simple name without parens
        TODO: AnnAssign(expr target, expr annotation, expr? value, int simple)
        """
        val = self.ast_eval(node.value)
        if len(node.targets) > 1: raise NotImplemented('Parallel assignments')
        n  = node.targets[0]
        self.symtable[n.id] = val

    def on_nameconstant(self, node):
        """
        NameConstant(singleton value)
        """
        return node.value

    def on_name(self, node):
        """
        Name(identifier id, expr_context ctx)
        """
        return self.symtable[node.id]

    def on_expr(self, node):
        """
        Expr(expr value)
        >>> i = Interpreter()
        >>> i.eval('123')
        123
        >>> i.eval('x = 123')
        """
        return self.ast_eval(node.value)

    def ast_eval(self, node):
        if node is None: return
        res = "on_%s" % node.__class__.__name__.lower()
        if hasattr(self, res): return getattr(self,res)(node)
        raise Exception('ast_eval: Not Implemented %s' % type(node))


    def on_compare(self, node):
        """
        Compare(expr left, cmpop* ops, expr* comparators)

        >>> i = Interpreter()
        >>> i.eval("2>3")
        False
        >>> i.eval("2<3")
        True
        """
        hd = self.ast_eval(node.left)
        op = node.ops[0]
        tl = self.ast_eval(node.comparators[0])
        return self.OpMap[type(op)](hd, tl)

    def on_unaryop(self, node):
        """
        >>> i = Interpreter()
        >>> i.eval("-2")
        -2
        """
        return self.OpMap[type(node.op)](self.ast_eval(node.operand))


    def on_binop(self, node):
        """
        >>> i = Interpreter()
        >>> i.eval("2+3")
        5
        """
        return self.OpMap[type(node.op)](self.ast_eval(node.left), self.ast_eval(node.right))

    def on_return(self, node):
        self._return = self.ast_eval(node.value)

    def on_if(self, node):
        body = node.body if self.ast_eval(node.test) else node.orelse
        self.body_eval(body)

    def on_call(self, node):
        func = self.ast_eval(node.func)
        args = [self.ast_eval(a) for a in node.args]
        return func(*args)

    def body_eval(self, stmts):
        v = None
        for n in stmts:
            v = self.ast_eval(n)
        return v


    def eval(self, src):
        """
        >>> i = Interpreter()
        >>> i.eval("x=1\\ny=2\\nx")
        1
        >>> i.eval("x = 2+3\\nx")
        5
        """
        try:
            node = self.parse(src)
            return self.ast_eval(node)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    def fetch_src(dt):
        return ''.join([e.source for es in dt for e in es.examples])
    import doctest, pudb
    finder = doctest.DocTestFinder()
    src = fetch_src(finder.find(Interpreter.eval))
    print("from interp import *\nimport pudb\npudb.set_trace()\n" + src)
