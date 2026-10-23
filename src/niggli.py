"""Niggli cell-reduction implementation derived from ``pyniggli``.

Copyright (c) 2017 Wiley Morgan
SPDX-License-Identifier: MIT

Upstream project: https://github.com/wsmorgan/pyniggli
The full MIT License is retained in ``LICENSES/MIT-pyniggli.txt``.

Contains the main class and functions for the Niggli reduced cell, as
described in the papers found at:
https://www.mendeley.com/viewer/?fileId=74bc20b9-a7a5-8e3d-4608-6ce09ea453e0&documentId=7b2a0aec-6dcf-3475-8834-96ddbb760220
https://www.mendeley.com/viewer/?fileId=39464089-ceb6-6b27-6e45-ade432b467cc&documentId=7dbd825c-282f-3f5a-99a3-cc89a1a5a11e
"""
from timeit import default_timer as timer
import numpy as np

from numba import njit


@njit(cache=True)
def niggli_check(A,B,C,xi,eta,zeta,eps):
        """Checks that the niggli reduced cell satisfies the niggli conditions.
        Conditions listed at: https://arxiv.org/pdf/1203.5146.pdf.
        Args:
            A (float): a.a
            B (float): b.b 
            C (float): c.c
            xi (float): 2*b.c
            eta (float): 2*c.a
            zeta (float): 2*a.b
        
        Returns:
            False if niggli conditons aren't met.
        """

        if not (A-eps > 0 and (A < B-eps or np.allclose(A,B,atol=eps)) and
                (B < C-eps or np.allclose(B,C,atol=eps))):
            return False
        
        if np.allclose(A,B,atol=eps) and not (abs(xi) < abs(eta)-eps or
                                              np.allclose(abs(xi),abs(eta),atol=eps)):
            return False
            
        if np.allclose(B,C,atol=eps) and not (abs(eta) < abs(zeta)-eps
                                              or np.allclose(abs(eta),abs(zeta),atol=eps)):
            return False

        if not ((xi-eps > 0 and eta-eps > 0 and zeta-eps > 0) or
                ((xi < 0-eps or np.allclose(xi,0,atol=eps))
                 and (eta < 0-eps or np.allclose(eta,0,atol=eps))
                 and (zeta < 0-eps or np.allclose(zeta,0,atol=eps)))):
            return False

        if not (abs(xi) < B-eps or np.allclose(abs(xi),B,atol=eps)):
            return False 
            
        if not ((abs(eta) < A-eps or np.allclose(abs(eta),A,atol=eps)) and (abs(zeta) < A-eps or
                                                               np.allclose(abs(zeta),A,atol=eps))):
            return False

        if not (C < A+B+C+xi+eta+zeta-eps or np.allclose(C, A+B+C+xi+eta+zeta,atol=eps)): 
            return False

        if np.allclose(xi,B,atol=eps) and not (zeta < 2.*eta-eps or
                                               np.allclose(zeta,2.*eta,atol=eps)):
            return False

        if np.allclose(eta,A,atol=eps) and not (zeta < 2.*xi-eps or
                                                np.allclose(zeta,2.*xi,atol=eps)):
            return False

        if np.allclose(zeta,A,atol=eps) and not (eta < 2.*xi-eps or
                                                 np.allclose(eta,2.*xi,atol=eps)):
            return False

        if np.allclose(xi,-B,atol=eps) and not np.allclose(zeta,0,atol=eps):
            return False

        if np.allclose(eta,-A,atol=eps) and not np.allclose(zeta,0,atol=eps):
            return False

        if np.allclose(zeta,-A,atol=eps) and not np.allclose(eta,0,atol=eps):
            return False

        if np.allclose(C,A+B+C+xi+eta+zeta,rtol=0.0) and not ((2.*A+2.*eta+zeta) < 0-eps or
                                                     np.allclose(2.*A+2.*eta+zeta,0,atol=eps)):
            return False

        return True

@njit()
def find_C3(l,m,n):
        """Finds the correct transformation matrix given the values of xi, eta, 
        and zeta for step 3.
        Args:
            l (int): Positive if xi is positive, negative if it isn't.
            m (int): Positive if eta is positive, negative if it isn't.
            n (int): Positive if zeta is positive, negative if it isn't.
        Returns:
            C (numpy ndarray): The transformation matrix.
        """

        i =1
        j = 1
        k = 1
        if l == (-1):
            i = -1
        if m == (-1):
            j = -1
        if n == (-1):
            k = -1

        C = np.array([[i,0,0],[0,j,0],[0,0,k]], np.float64)
            
        return C

@njit()
def find_C4(l,m,n):
        """Finds the correct transformation matrix given the values of xi, eta, 
        and zeta for step 4.
        Args:
            l (int): Positive if xi is positive, negative if it isn't.
            m (int): Positive if eta is positive, negative if it isn't.
            n (int): Positive if zeta is positive, negative if it isn't.
        Returns:
            C (numpy ndarray): The transformation matrix.
        """

        i = 1
        j = 1
        k = 1

        if l == (-1) and m == (-1) and n == (-1):
            C = np.array([[i,0,0],[0,j,0],[0,0,k]], np.float64)
            return C
        
        r = -1
        if l == 1:
            i = -1
        elif l == 0:
            r = 0

        if m == 1:
            j = -1
        elif m == 0:
            r = 1

        if n == 1:
            k = -1
        elif n == 0:
            r = 2

        if i*j*k == (-1):
            if r==0:
                i = -1
            elif r==1:
                j = -1
            elif r==2:
                k = -1
                
        C = np.array([[i,0,0],[0,j,0],[0,0,k]], np.float64)

        return C 

@njit()
def get_fast_params(original, C_mat, eps):

    mat = np.dot(original,C_mat)

    a = np.ascontiguousarray(mat[:,0])
    b = np.ascontiguousarray(mat[:,1])
    c = np.ascontiguousarray(mat[:,2])

    A = np.dot(a,a)
    B = np.dot(b,b)
    C = np.dot(c,c)
    xi = 2.0*np.dot(b,c)
    eta = 2.0*np.dot(c,a)
    zeta = 2.0*np.dot(a,b)
    l, m, n = 0, 0, 0
    if xi<-eps:
        l = -1
    elif xi> eps:
        l = 1
        
    if eta<-eps:
        m = -1
    elif eta> eps:
        m = 1
            
    if zeta<-eps:
        n = -1
    elif zeta> eps:
        n = 1

    return A, B, C, xi, eta, zeta, l, m, n

@njit()
def fast_niggli_reduction_core(A, B, C, xi, eta, zeta, l, m, n, eps,
                               original, C_mat):
    count = 0
    reduced = False
    
    while not reduced and count <=1000:
            
        reduced = True
        count += 1
        #1
        if A > (B+eps) or (not (abs(A-B)>eps) and abs(xi)>(abs(eta)+eps)):
            C_mat = np.dot(C_mat,np.array([[0.,-1.,0.],[-1.,0.,0.],[0.,0.,-1.]], np.float64))
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            reduced = False
            
        #2
        if B > (C+eps) or (not (abs(C-B)>eps) and abs(eta)>abs(zeta)+eps):
            C_mat = np.dot(C_mat,np.array([[-1.,0.,0.],[0.,0.,-1.],[0.,-1.,0.]],np.float64))
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            reduced = False
            continue
            #go to 1
        #3
        if l*m*n==1:
            M = find_C3(l,m,n)
            C_mat = np.dot(C_mat,M)
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            if not np.allclose(M,np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float64)):
                reduced = False
            
        #4
        if l*m*n == 0 or l*m*n == (-1):
            M = find_C4(l,m,n)
            C_mat = np.dot(C_mat,M)
            if not np.allclose(M,np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float64)):
                reduced = False
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            
        #5
        if abs(xi)>(B+eps) or (not (abs(B-xi)>eps) and (2*eta<(zeta-eps))) or (not abs(B+xi)>eps and zeta<(-eps)):
            C_mat = np.dot(C_mat,np.array([[1.,0.,0.],[0.,1.,-np.sign(xi)],[0.,0.,1.]], np.float64))
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            reduced = False
            continue
            #go to 1
        #6
        if abs(eta)>(A+eps) or (not abs(A-eta)>eps and (2*xi<(zeta-eps))) or (not abs(A+eta)>eps and zeta<(-eps)):
            C_mat = np.dot(C_mat,np.array([[1.,0.,-np.sign(eta)],[0.,1.,0.],[0.,0.,1.]], np.float64))
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            reduced = False
            continue
            #go to 1
        #7
        if abs(zeta)>(A+eps) or (not abs(A-zeta)>eps and (2*xi<(eta-eps))) or (not abs(A+zeta)>eps and eta<(-eps)):
            C_mat = np.dot(C_mat,np.array([[1.,-np.sign(zeta),0.],[0.,1.,0.],[0.,0.,1.]], np.float64))
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            reduced = False
            continue
            #go to 1
        #8
        if xi+eta+zeta+A+B<(-eps) or (not abs(xi+eta+zeta+A+B)>eps and (2*(A+eta)+zeta)>eps):
            C_mat = np.dot(C_mat,np.array([[1.,0.,1.],[0.,1.,1.],[0.,0.,1.]], np.float64))
            A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
            reduced = False
            continue
            #go to 1
    return A, B, C, xi, eta, zeta, C_mat

def fast_niggli_reduction(original):
    volume = abs(np.linalg.det(original))
    eps = (1e-5)*volume**(1./3.) #(1e-5)

    count, reduced, C_mat = 0, False, np.array([[1,0,0],[0,1,0],[0,0,1]], np.float64)
    A, B, C, xi, eta, zeta, l, m, n = get_fast_params(original, C_mat, eps)
    
    A, B, C, xi, eta, zeta, C_final = fast_niggli_reduction_core(A, B, C, xi, eta, zeta, l, m, n,
                                                        eps, original, C_mat)
    is_niggli_reduced =  niggli_check(A,B,C,xi,eta,zeta,eps)
    reduced_cell = np.dot(original,C_final)
    return reduced_cell, [A, B, C, xi, eta, zeta], C_final,is_niggli_reduced 




def test_main():
    #,b,c = 7.81, 7.81, 7.81
    #alpha, beta, gamma = np.deg2rad(109.13), np.deg2rad(109.13), np.deg2rad(109.13)
    #a,b,c = 9.2536, 9.2536, 6.7028
    #alpha,beta, gamma = np.deg2rad(90), np.deg2rad(90), np.deg2rad(120)
    #a,b,c = 5.06, 8.08, 8.87
    #alpha, beta, gamma = np.deg2rad(91.5), np.deg2rad(93.2), np.deg2rad(94.2)
    a,b,c = 5.05, 11.83, 12.18
    alpha, beta, gamma = np.deg2rad(95.3), np.deg2rad(90.5), np.deg2rad(95.2)
    v1 = np.array([a, 0, 0])
    v2 = np.array([b*np.cos(gamma), b*np.sin(gamma), 0.0])
    cosalphastar = (np.cos(beta)*np.cos(gamma) - np.cos(alpha))/(np.sin(beta)*np.sin(gamma))
    sinalphastar = np.sqrt(1 - cosalphastar**2)
    v3 = np.array([c*np.cos(beta), -c*np.sin(beta)*cosalphastar, c*np.sin(beta)*sinalphastar])

    A = np.transpose([v1,v2,v3])
    print("[v1,v2,v3] = ")
    print(A)
    niggli_cell, _, _. _ = fast_niggli_reduction(A)
    # The niggli cell can be retrieved with
    print("niggli cell = ")
    print(niggli_cell)
    v1_red, v2_red, v3_red = niggli_cell[:,0], niggli_cell[:,1],niggli_cell[:,2]
    a_red = np.linalg.norm(v1_red)
    b_red = np.linalg.norm(v2_red)
    c_red = np.linalg.norm(v3_red)
    alpha_red = np.rad2deg(np.arccos(np.dot(v2_red, v3_red)/(b_red*c_red)))
    beta_red = np.rad2deg(np.arccos(np.dot(v1_red, v3_red)/(a_red*c_red)))
    gamma_red = np.rad2deg(np.arccos(np.dot(v1_red, v2_red)/(a_red*b_red)))
    print(f"a={a_red}, b={b_red}, c={c_red}, alpha={alpha_red}, beta={beta_red}, gamma={gamma_red}")
    #array([[-0.5  0.  -0.5],[-0.5  0.   0.5],[ 0.  -3.   0. ]])

#test_main()

@njit(cache=True, error_model="numpy")
def _cell_vectors(a, b, c, alpha, beta, gamma):
    """Construct the original float64 column basis without per-cell NumPy dispatch."""
    cosalphastar = (np.cos(beta)*np.cos(gamma) - np.cos(alpha))/(np.sin(beta)*np.sin(gamma))
    if 1 - cosalphastar**2 < 0.0:
        return np.zeros((3, 3), np.float64).T, False
    sinalphastar = np.sqrt(1 - cosalphastar**2)
    # The old np.transpose([v1, v2, v3]) also produces a float64 Fortran-layout
    # matrix, including when the incoming cell parameters are float32.
    basis = np.zeros((3, 3), np.float64).T
    basis[0, 0] = a
    basis[0, 1] = b*np.cos(gamma)
    basis[1, 1] = b*np.sin(gamma)
    basis[0, 2] = c*np.cos(beta)
    basis[1, 2] = -c*np.sin(beta)*cosalphastar
    basis[2, 2] = c*np.sin(beta)*sinalphastar
    return basis, True


@njit(cache=True, error_model="numpy")
def _reduced_cell_parameters(niggli_cell):
    """Recover lengths and angles with the existing formulas and precision."""
    # Each column is a vector; one 3x3 layout conversion avoids separate
    # strided-vector copies in the compiled norms and dot products.
    niggli_cell = np.asfortranarray(niggli_cell)
    v1_red, v2_red, v3_red = niggli_cell[:,0], niggli_cell[:,1], niggli_cell[:,2]
    a_red = np.linalg.norm(v1_red)
    b_red = np.linalg.norm(v2_red)
    c_red = np.linalg.norm(v3_red)
    alpha_red = np.rad2deg(np.arccos(np.dot(v2_red, v3_red)/(b_red*c_red)))
    beta_red = np.rad2deg(np.arccos(np.dot(v1_red, v3_red)/(a_red*c_red)))
    gamma_red = np.rad2deg(np.arccos(np.dot(v1_red, v2_red)/(a_red*b_red)))
    return [a_red, b_red, c_red, alpha_red, beta_red, gamma_red]


def cell_to_niggli(cell):
    # Keep degree conversion in the input scalar precision, as before. The
    # reduction core, its volume-scaled tolerance, and sentinel are unchanged.
    a, b, c = cell[0], cell[1], cell[2]
    alpha, beta, gamma = np.deg2rad(cell[3]), np.deg2rad(cell[4]), np.deg2rad(cell[5])
    original, valid = _cell_vectors(a, b, c, alpha, beta, gamma)
    if not valid:
        return [1000, 1000, 1000, 90, 90, 90]
    try:
        niggli_cell, _, _, _ = fast_niggli_reduction(original)
        return _reduced_cell_parameters(niggli_cell)
    except RuntimeError:
        return [1000, 1000, 1000, 90, 90, 90]
