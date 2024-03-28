# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:57:13 2024

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:41:13 2024

@author: Zeferino Parada García
     Optimización Numérica
           ITAM
"""

def qp_directo(Q,A,c,b):
    import numpy as np
    # Método directo para  Programación
    # Cudrática Convexa
    # Min     (0.5)*x.T * Q *x + c.T*x
    # S.A.       A*x =  b
    #
    #  Q.- matriz nxn  simétrica y positiva definida
    # A.- matriz mxn tal que rango(A) = m
    # c.- vector de nx1
    #  b.- vector de mx1
    #------------------------------------------------
    
    (m,n) = np.shape(A)
    K = np.concatenate((Q, A.T),1)
    M = np.zeros((m,m))
    T = np.concatenate((A, M),1)
    K = np.concatenate((K,T),0)
    ld = np.concatenate((-c, b),0)
    w = np.linalg.solve(K,ld)
    x = w[0:n]
    y = w[(n+1):(n+m)]
    return x, y
   # Fin de qp_directo
   #----------------------------------
   
   
   
   
def mi_esp_nulo(Q,A,c,b):
       # Método del espacio nulo para el problema cuadrático convexo
       # Min (1/2)*x.T*q*x + c.T*x
       # Sujeto a  A*x = b
       #--------------------------------------------------------
       import numpy as np
       (m,n) = np.shape(A)
       #---- Descomposición en valores singulares de A --------
       (U,S,Vh) = np.linalg.svd(A, full_matrices=True)
       V = Vh.T    
       V1 = V[:,0:m]
       #-----base del espacio nulo---
       Z = V[:,m:n]
       # --------Solución particular/ A*xpar = b
       xpar = np.dot(U.T,b)
       Sinv = 1/S
       xpar = Sinv*xpar
       xpar = np.dot(V1,xpar)
       #--------------------------
       # matriz del problema cuadrático sin restricciones
       QZ = np.dot(Z.T,Q)
       QZ = np.dot(QZ,Z)
       #----------------------------------------
       #  lado derecho del sistema
       ld = np.dot(Q,xpar)+c
       ld = -np.dot(Z.T,ld)
       #------------------------
       # Solución del problema cuadrático sin restricciones
       xz = np.linalg.solve(QZ,ld)
      
       #------------------------------
       # Solución del problema original
       xstar = xpar + np.dot(Z,xz)
       
       return xstar

   #  Fin de mi_esp_nulo,py
   #---------------------------------------------------
   
   

def paso_intpoint(u,v):
    # Recorte de paso para el método de puntos interiores
    # para Programación Cuadrática.
    # u es un vector de dimensión p tal que u[i]>0 para todo i
    # v es un vector de dimensión p.
    # Construir un escalar alfa <= 1 tal que
    # u +(alfa)*v >=0.
    # Si v[i] >= 0 para toda i
    # el escalar alfa = 1.
    # En caso contrario existe un índice i tal que v[i] <0.
    #------------------------------------------------
    # Optimización Numérica
    #   ITAM
    #    20 de marzo de 2024
    #-----------------------------------------
    import numpy as np
    p = len(u)
    v_alfa = np.ones(p)
    for i in range(p):
         if(v[i]<0):
             v_alfa[i] = -u[i]/v[i]
             #---Fin de if---
             
     #----------Fin de for -----------

    alfa = np.amin(v_alfa)
    alfa = np.amin([alfa, 1.0])        
    return alfa
    #------- Fin de paso:intpoint.py-----------------
    
    
    
    
    
def myqp_intpoint(Q, A, F,c,b,d):
    # Método de Puntos Interiores para Programación
    # Cuadrática
    # Min (1/2) x.T*Q*x + c.T*x
    # s.a.   A*x = b
    #        F*x >= d
    # Q es simétrica positiva definda de orden n
    # A es matriz mxn tal que rango(A) = m.
    # F matriz pxn.
    # c vector de dimensión n
    # b vector de dimensión m
    # d vector de dimensión 
    #
    # return
    # x vector de dimensión n con el mínimo del problema
    # y vector de dimensión m y es el multiplicador de
    #    Lagrange para: A*x . b = 0
    # mu vector de dimensión p y es el multiplicador de Lagrange
    #    para_ f*x -d >= 0
    # z vector de dimensión p tal que z = F*x-d 
    # iter número de iteraciones que usamos
    #-----------------------------------------------
    # Optimazación Numérica
    #   ITAM
    # 20 de marzo de 2024
    #-----------------------------------------------
    import numpy as np
    import copy
    from myqp import paso_intpoint
    
    #---------Valores Iniciales---------
    n = len(c)
    m = len(b)
    p = len(d)
    tol = 10**(-5)
    maxiter = 100
    iter = 0
    #-------------Vectores iniciales-----------
    x = np.ones(n)
    y = np.zeros(m)
    mu= np.ones(p)
    z = np.ones(p)
    #--------------------------------
    # lado derecho
    tau = 1/2
    v1 = np.dot(Q,x)+np.dot(A.T, y)-np.dot(F.T,mu) +c
    v2 = np.dot(A,x)-b
    v3 = -np.dot(F,x) +z +d
    v4 = np.multiply(mu,z)

    cnpo = np.concatenate((v1,v2),0)
    cnpo = np.concatenate((cnpo, v3),0)
    cnpo = np.concatenate((cnpo, v4),0)
    norma_cnpo = np.linalg.norm(cnpo)
    #--------------------------------------------
    # Proceso iterativo
    while(norma_cnpo > tol and iter < maxiter):
        cnpo_pert = copy.copy(cnpo)
        cnpo_pert[n+m+p:n+m+p+p]= cnpo_pert[n+m+p:n+m+p+p]-tau
        #---------------------------------------
        #  Matriz de Newton
        dim = n + m + p +p
        M = np.zeros((dim,dim))
        M[0:n, 0:n]  = Q
        M[0:n, n:(n+m)]= A.T
        M[0:n, (n+m):(n+m+p)]=-F.T
        M[n:n+m, 0:n]= A
        M[(n+m):(n+m+p), 0:n]= -F
        M[(n+m):(n+m+p), (n+m+p):(n+m+p+p)]= np.identity(p)
        M[(n+m+p):dim, (n+m):(n+m+p)]= np.diag(z)
        M[(n+m+p):dim, (n+m+p):dim]= np.diag(mu)
        #-------------------------------------
        # Solución del sistema lineal
        dw = np.linalg.solve(M,-cnpo_pert)
        dx = dw[0:n]
        dy = dw[n:n+m]
        dmu =dw[n+m:n+m+p]
        dz =dw[n+m+p:dim]
        #-----------------------------------
        alfa1 = paso_intpoint(mu,dmu)
        alfa2 = paso_intpoint(z,dz)
        alfa = (0.95)*np.amin([alfa1, alfa2, 1.0])
        #---------------------------------
        # Actualizamos los vectores-----
        x =  x  + alfa*dx
        y =  y  + alfa*dy
        mu = mu + alfa*dmu
        z =  z  + alfa*dz
        #--------------------------------
        iter = iter +1
        tau = np.dot(mu,z)/(2*p)
        #------------------------------------
        v1 = np.dot(Q,x)+np.dot(A.T, y)-np.dot(F.T,mu) +c
        v2 = np.dot(A,x)-b
        v3 = -np.dot(F,x) +z +d
        v4 = np.multiply(mu,z)

        cnpo = np.concatenate((v1,v2),0)
        cnpo = np.concatenate((cnpo, v3),0)
        cnpo = np.concatenate((cnpo, v4),0)
        norma_cnpo = np.linalg.norm(cnpo)
        
        print("iter=", iter,"|","||cnpo||=",norma_cnpo)
        #----------------------------------
        if(norma_cnpo <=tol or iter ==maxiter):
           return x,y,mu,z,iter
           break
       
        #  ---- Fin de myqp_intpoint.py----
        #------------------------------------


def myqp_intpoint_proy(Q,F,c,d,verbose=True):
    # Método de Puntos Interiores para Programación
    # Cuadrática
    # Min (1/2) x.T*Q*x + c.T*x
    # s.a.   A*x = b
    #        F*x >= d
    # Q es simétrica positiva definda de orden n
    # A es matriz mxn tal que rango(A) = m.
    # F matriz pxn.
    # c vector de dimensión n
    # b vector de dimensión m
    # d vector de dimensión 
    #
    # return
    # x vector de dimensión n con el mínimo del problema
    # y vector de dimensión m y es el multiplicador de
    #    Lagrange para: A*x . b = 0
    # mu vector de dimensión p y es el multiplicador de Lagrange
    #    para_ f*x -d >= 0
    # z vector de dimensión p tal que z = F*x-d 
    # iter número de iteraciones que usamos
    #-----------------------------------------------
    # Optimazación Numérica
    #   ITAM
    # 20 de marzo de 2024
    #-----------------------------------------------
    import numpy as np
    import copy
    from myqp import paso_intpoint
    
    #---------Valores Iniciales---------
    n = len(c)
    p = len(d)
    tol = 10**(-5)
    maxiter = 100
    iter = 0
    #-------------Vectores iniciales-----------
    x = np.ones(n)
    mu= np.ones(p)
    z = np.ones(p)
    #--------------------------------
    # lado derecho
    tau = 1/2
    v1 = np.dot(Q,x)-np.dot(F.T,mu) + c
    v3 = -np.dot(F,x) + z + d
    v4 = np.multiply(mu,z)

    cnpo = np.concatenate((v1,v3,v4))
    norma_cnpo = np.linalg.norm(cnpo)
    #--------------------------------------------
    # Proceso iterativo
    while(norma_cnpo > tol and iter < maxiter):
        cnpo_pert = copy.copy(cnpo)
        cnpo_pert[n+p:n+p+p]= cnpo_pert[n+p:n+p+p]-tau
        #---------------------------------------
        #  Matriz de Newton
        dim = n + p + p
        M = np.zeros((dim,dim))
        M[0:n, 0:n]  = Q        
        M[0:n, n:n+p]=-F.T
        M[n:n+p, 0:n]= -F
        M[n:n+p, n+p:n+p+p]= np.identity(p)
        M[n+p:dim, n:n+p]= np.diag(z)
        M[n+p:dim, n+p:dim]= np.diag(mu)
        #-------------------------------------
        # Solución del sistema lineal
        dw = np.linalg.solve(M,-cnpo_pert)
        dx = dw[0:n]
        dmu =dw[n:n+p]
        dz =dw[n+p:dim]
        #-----------------------------------
        alfa1 = paso_intpoint(mu,dmu)
        alfa2 = paso_intpoint(z,dz)
        alfa = (0.95)*np.amin([alfa1, alfa2, 1.0])
        #---------------------------------
        # Actualizamos los vectores-----
        x =  x  + alfa*dx
        mu = mu + alfa*dmu
        z =  z  + alfa*dz
        #--------------------------------
        iter = iter +1
        tau = np.dot(mu,z)/(2*p)
        #------------------------------------
        v1 = np.dot(Q,x)-np.dot(F.T,mu) + c
        v3 = -np.dot(F,x) + z + d
        v4 = np.multiply(mu,z)

        cnpo = np.concatenate((v1,v3,v4))
        norma_cnpo = np.linalg.norm(cnpo)
        
        if verbose:
            print("iter=", iter,"|","||cnpo||=",norma_cnpo)
        #----------------------------------
        if(norma_cnpo <=tol or iter ==maxiter):
           return x,mu,z,iter
       
        #  ---- Fin de myqp_intpoint_proy.py----
        #------------------------------------