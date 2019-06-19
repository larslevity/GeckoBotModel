# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:53:17 2019

@author: AmP
"""

from sympy import symbols, exp, I, sin, cos
import sympy

#sympy.init_printing()


a0, a1, a2 = symbols('a_0 a_1 a_2', real=True)
b0, b1, b2 = symbols('b_0 b_1 b_2', real=True)
c0, c1, c2, c3 = symbols('c_0 c_1 c_2 c_3', real=True)


#def dx(x1, x2):
#    return [.02*x1 + .13*x2 - .47*x2**2,
#            -(.07*x2 - .29*x2**2 + .02*x1*x2)]
#def deps(x1, x2):
#    return (-.005*x1 - 10.85*x2 - 2.55*x2**2 - .835*x1*x2)


#def dx(x1, x2):
#    return [a0*x1 + a1*x2 + a2*x2**2,
#            b0*x2 + b1*x2**2 + b2*x1*x2]
#
#
#def deps(x1, x2):
#    return (c0*x1 + c1*x2 + c2*x2**2 + c3*x1*x2)


def dx(x1, x2):
    return [.02*x1 + .13*x2 - .47*x2**2,
            -(.07*x2 - .29*x2**2 + .02*x1*x2)]

def deps(x1, x2):
    return (-.005*x1 - 10.85*x2 - 2.55*x2**2 - .835*x1*x2)


def sumsin(x, n):
    return 1/sin(x/2)*sin((n+1)*x/2)*sin(n*x/2)

def sumcos(x, n):
    return 1/sin(x/2)*sin((n+1)*x/2)*cos(n*x/2)

def Esumsin(x, n):
    return (2*I)/(exp(I*x/2)-exp(-I*x/2)) * ((exp(I*((n+1)*x)/2) - exp(-I*((n+1)*x)/2))/(2*I)) * ((exp(I*(n*x/2)) - exp(-I*(n*x/2)) )/(2*I))

def Esumcos(x, n):
    return (2*I)/(exp(I*x/2)-exp(-I*x/2)) * ((exp(I*((n+1)*x)/2) - exp(-I*((n+1)*x)/2))/(2*I)) * ((exp(I*(n*x/2)) + exp(-I*(n*x/2)) )/(2))

def R(alp):
    return [[cos(alp), -sin(alp)],
            [sin(alp), cos(alp)]]

def Esin(alp):
    return (exp(I*alp) - exp(-I*alp))/(2*I)


def Ecos(alp):
    return (exp(I*alp) + exp(-I*alp))/(2)


def ER(alp):
    return [[Ecos(alp), -Esin(alp)],
            [Esin(alp), Ecos(alp)]]


def sumR(alp, n):
    return [[sumcos(alp, n), -sumsin(alp, n)],
            [sumsin(alp, n), sumcos(alp, n)]]

def EsumR(alp, n):
    return [[Esumcos(alp, n), -Esumsin(alp, n)],
            [Esumsin(alp, n), Esumcos(alp, n)]]


def multiply(A, x):
    a11 = A[0][0]
    a12 = A[0][1]
    a21 = A[1][0]
    a22 = A[1][1]
    return [a11*x[0] + a12*x[1], a21*x[0] + a22*x[1]]



def calc_d(xbar, dx, deps, n):
    val1 = multiply(R(-n*deps), xbar)
    val2 = multiply(sumR(-deps, n), dx)
    dist_vec_x = val1[0] - val2[0]
    dist_vec_y = val1[1] - val2[1]
    return dist_vec_x**2 + dist_vec_y**2


def Ecalc_d(xbar, dx, deps, n):
    val1 = multiply(ER(-n*deps), xbar)
    val2 = multiply(EsumR(-deps, n), dx)
    dist_vec_x = val1[0] - val2[0]
    dist_vec_y = val1[1] - val2[1]
    return dist_vec_x**2 + dist_vec_y**2




x1 = symbols('x1', real=True)
x2 = symbols('x2', real=True)
n = symbols('n', real=True)
#n = 2
deps_dummy = symbols('delta_varepsilon', real=True)



# %% Euler sum Sin diff
#x = symbols('delta_varepsilon', real=True)
x = deps(x1, x2)


#sumsin = (2*I)/(exp(I*x/2)-exp(-I*x/2)) * ((exp(I*((n+1)*x)/2) - exp(-I*((n+1)*x)/2))/(2*I)) * ((exp(I*(n*x/2)) - exp(-I*(n*x/2)) )/(2*I))
#print('sumsin1')
#sympy.print_latex(sympy.simplify(sumsin))
#print('diff sumsin1 dx1')
#sympy.print_latex(sympy.simplify(sympy.diff(sumsin, x1)))
#print('diff sumsin1 dx2')
#sympy.print_latex(sympy.simplify(sympy.diff(sumsin, x2)))

# %% Euler SumCos
#x = Esumcos(deps_dummy, n)
#print('SumCos Euler:')
#sympy.print_latex(sympy.simplify(x))
#print('diff SumCos Euler:')
#sympy.print_latex(sympy.simplify(sympy.diff(x, deps_dummy)))

# %% Normal SumSin Diff
#sumsin2 = 1/sin(x/2)*sin((n+1)*x/2)*sin(n*x/2)
#sumsin2 = sympy.simplify(sumsin2)
#
#print('sumsin2')
#sympy.print_latex(sympy.simplify(sumsin2))
#
#print('diff sumsin2 dx1')
#sympy.print_latex(sympy.simplify(sympy.diff(sumsin2, x1)))
#print('diff sumsin2 dx2')
#sympy.print_latex(sympy.simplify(sympy.diff(sumsin2, x2)))

# %% Normal SumCos Diff
x = sumcos(deps_dummy, n)
print('SumCos:')
sympy.print_latex(sympy.simplify(x))
print('diff SumCos:')
sympy.print_latex(sympy.simplify(sympy.diff(x, deps_dummy)))



# %% Sym Diff d
xbarx = symbols('xb_x', real=True)
xbary = symbols('xb_y', real=True)
xbar = [xbarx, xbary]

# %% Distance Euler Formulation

d = Ecalc_d(xbar, dx(x1, x2), deps(x1, x2), n)

print('Distance to goal -- Euler')
sympy.print_latex(sympy.simplify(d))


# %% Distance
n = 2

d = calc_d(xbar, dx(x1, x2), deps(x1, x2), n)

print('Distance to goal')
sympy.print_latex(d)


# %% Simplify
print('Distance to goal - symplified')
sympy.print_latex(sympy.simplify(d))
