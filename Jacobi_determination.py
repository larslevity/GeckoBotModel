# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:53:17 2019

@author: AmP
"""

from sympy import symbols, exp, I, sin, cos
import sympy

sympy.init_printing()


a0, a1, a2 = symbols('a_0 a_1 a_2', real=True)
b0, b1, b2 = symbols('b_0 b_1 b_2', real=True)
c0, c1, c2, c3 = symbols('c_0 c_1 c_2 c_3', real=True)


#def dx(x1, x2):
#    return [.02*x1 + .13*x2 - .47*x2**2,
#            -(.07*x2 - .29*x2**2 + .02*x1*x2)]
#def deps(x1, x2):
#    return (-.005*x1 - 10.85*x2 - 2.55*x2**2 - .835*x1*x2)


def dx(x1, x2):
    return [a0*x1 + a1*x2 + a2*x2**2,
            b0*x2 + b1*x2**2 + b2*x1*x2]


def deps(x1, x2):
    return (c0*x1 + c1*x2 + c2*x2**2 + c3*x1*x2)


x1 = symbols('x1', real=True)
x2 = symbols('x2', real=True)
n = symbols('n', real=True)


#x = symbols('delta_varepsilon', real=True)
x = deps(x1, x2)


sumsin = (2*I)/(exp(I*x/2)-exp(-I*x/2)) * ((exp(I*((n+1)*x)/2) - exp(-I*((n+1)*x)/2))/(2*I)) * ((exp(I*(n*x/2)) - exp(-I*(n*x/2)) )/(2*I))

print('sumsin1')
sympy.print_latex(sympy.simplify(sumsin))
print('diff sumsin1 dx1')
sympy.print_latex(sympy.simplify(sympy.diff(sumsin, x1)))
print('diff sumsin1 dx2')
sympy.print_latex(sympy.simplify(sympy.diff(sumsin, x2)))


sumsin2 = 1/sin(x/2)*sin((n+1)*x/2)*sin(n*x/2)
sumsin2 = sympy.simplify(sumsin2)

print('sumsin2')
sympy.print_latex(sympy.simplify(sumsin2))

print('diff sumsin2 dx1')
sympy.print_latex(sympy.simplify(sympy.diff(sumsin2, x1)))
print('diff sumsin2 dx2')
sympy.print_latex(sympy.simplify(sympy.diff(sumsin2, x2)))

