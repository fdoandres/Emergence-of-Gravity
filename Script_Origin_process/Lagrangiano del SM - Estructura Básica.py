import sympy as sp
from sympy import symbols, Derivative, I, exp, sqrt, Matrix

class StandardModelLagrangian:
    def __init__(self):
        # Campos gauge
        self.B_mu, self.W_mu, self.G_mu = symbols('B_mu W_mu G_mu')
        
        # Campo de Higgs
        self.phi, self.phi_dagger = symbols('phi phi_dagger')
        
        # Campos fermiónicos
        self.psi_L, self.psi_R = symbols('psi_L psi_R')
        
        # Constantes de acoplamiento
        self.g, self.gp, self.gs = symbols('g g\' g_s')
        self.yt, self.lambd = symbols('y_t lambda')
        
        # Tensores de campo
        self.B_munu, self.W_munu, self.G_munu = symbols('B_{mu nu} W_{mu nu} G_{mu nu}')
    
    def gauge_sector(self):
        """Sector gauge puro"""
        L_gauge = -1/4 * (self.B_munu**2 + self.W_munu**2 + self.G_munu**2)
        return L_gauge
    
    def higgs_sector(self):
        """Sector de Higgs convencional"""
        D_mu_phi = Derivative(self.phi) - I*(self.g/2)*self.W_mu*self.phi - I*(self.gp/2)*self.B_mu*self.phi
        L_higgs = (D_mu_phi).conjugate() * D_mu_phi - self.lambd*(self.phi.conjugate()*self.phi - 246**2)**2
        return L_higgs
    
    def yukawa_sector(self):
        """Sector de Yukawa"""
        L_yukawa = -self.yt * (self.psi_L.conjugate() * self.phi * self.psi_R + 
                              self.psi_R.conjugate() * self.phi.conjugate() * self.psi_L)
        return L_yukawa
    
    def full_SM_lagrangian(self):
        """Lagrangiano completo del Modelo Estándar"""
        return self.gauge_sector() + self.higgs_sector() + self.yukawa_sector()