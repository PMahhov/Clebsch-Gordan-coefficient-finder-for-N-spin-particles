#============================================================================
# Name                      : cg_coefs_Mahhov_Pokal.py
# Code, Algorithms, Method  : Peter Mahhov and Vandan Pokal
# Date Created              : 12.12.2020
# Description               : Calculating Clebsh-Gordan Coefficients for a set of particles with spins
#============================================================================

import numpy as np
import scipy as sp
import cmath
import time


# Initial function, starting user interface:
def ask_for_spins():
  print("Please input the state:")
  while(True):
    particles = input("What is the number of particles? \n")
    if particles.isnumeric():
      if int(particles) > 1:
        particles = int(particles)
        break
    print("Please input a whole number larger than 1")
  spins = []
  for i in range(particles):
      print("Particle number",i+1)
      spin = (input("Input the spin of the particle: "))
      spin = float(spin)
      spins.append(spin)
  return spins

spins = ask_for_spins()

# Kroenecker product
def kr_product(A, B, dtype = complex):    # inputs A and B are np matrices
  return np.kron(A,B)


# Input s values, output list of m values
def get_m(s):
  M = []
  i = s
  while(i >= -s):
    M.append(i)
    i = i-1
  return M

# Formatting:
#print(get_m(1/2))
#print(get_m(1))
#get_m(3/2)

def get_Sz(s):
  M = get_m(s)
  Sz = np.zeros((len(M),len(M)))
  for i in range(len(M)):
    Sz[i,i] = M[i]
  return Sz

#Format:
#get_Sz(3/2)

def getC_plus(s,m):
  return cmath.sqrt((s-m)*(s+m+1))

def getC_minus(s,m):
  return cmath.sqrt((s+m)*(s-m+1))

def get_S_plus(s):
  M = get_m(s)
  S = np.zeros((len(M),len(M)), dtype=complex)
  for i in range(len(M)-1):
    S[i,i+1] = getC_plus(s,M[i+1])
  return S

def get_S_minus(s):
  M = get_m(s)
  S = np.zeros((len(M),len(M)), dtype=complex)
  for i in range(len(M)-1):
    S[i+1, i] = getC_minus(s,M[i])
  return S

def get_S_x(s):
  S_plus = get_S_plus(s)
  S_minus = get_S_minus(s)
  return 0.5*(S_plus + S_minus)

def get_S_y(s):
  S_plus = get_S_plus(s)
  S_minus = get_S_minus(s)
  return -0.5j*(S_plus - S_minus)


def eigenvectors(matrix):
  D,v = np.linalg.eig(matrix)
  v = np.asmatrix(v)
  return v

def seq_kr_of_Sz_evectors(spins=spins):     # Provides the sequential kroenecker product of the eigenvectors of Sz

  # eigenvectors of Sz matrix
  Sz_matrices = []

  for i in range(len(spins)):
    Sz_matrices.append(get_Sz(spins[i]))

  e_matrices = []

  for j in Sz_matrices:
    e_matrices.append(eigenvectors(j))


  # sequential kroenecker: take the kr_product of two values, then the kr_product of the result with a third value, etc until you run out of values
  if len(spins) > 1:
    s = np.asmatrix(e_matrices[0][0]).H
    for i in range(len(spins)-1):
      s = kr_product(s, np.asmatrix(e_matrices[i+1][0]).H)
  return s


def correct_S(value, base):
  if base=="x":
    return get_S_x(value)
  elif base=="y":
    return get_S_y(value)
  elif base=="z":
    return get_Sz(value)
  elif base=="plus" or base=="+":
    return get_S_plus(value)
  elif base=="minus" or base=="-":
    return get_S_minus(value)
  else:
    raise Exception("wrong base")

def comb_lowering_matrix_addends(base, spins=spins):
  addends = []
  for i in range(len(spins)): # for each location of S
    if i == 0:
      product = correct_S(spins[0],base)
    else:
      product = np.identity(len(correct_S(spins[0],base)))

    for j in range(len(spins)-1): # number of kroenecker products done to get one addend
      if j+1 == i:
        product = kr_product(product, correct_S(spins[j+1],base))
      else:
        product = kr_product(product, np.identity(len(correct_S(spins[j+1],base))))
    addends.append(product)
  return addends

def comb_lowering_matrix(base,spins=spins):
  sum = 0
  addends = comb_lowering_matrix_addends(base,spins)
  for i in range(len(spins)):
    sum += addends[i]
  return sum


def get_max_spin_space(spins=spins):
  normalized_state_vectors = [seq_kr_of_Sz_evectors(spins)]


  state_vector = np.dot(comb_lowering_matrix("minus",spins),seq_kr_of_Sz_evectors(spins))
  norm = np.linalg.norm(state_vector)

  normalized_state_vector = state_vector/norm
  normalized_state_vectors.append(normalized_state_vector)

  while(np.count_nonzero(normalized_state_vector) != 0):

    state_vector = np.dot(comb_lowering_matrix("minus",spins),normalized_state_vector)
    norm = np.linalg.norm(state_vector)

    if (np.count_nonzero(state_vector) == 0):
      break

    normalized_state_vector = state_vector/norm
    normalized_state_vectors.append(normalized_state_vector)

  maximum_spin_space = np.zeros((len(normalized_state_vectors),len(normalized_state_vectors[0])),dtype=complex)

  for i in range(len(normalized_state_vectors)):
    for j in range(len(normalized_state_vectors[0])):
      maximum_spin_space[i][j] = normalized_state_vectors[i][j]
  maximum_spin_space = np.asmatrix(maximum_spin_space).H
  return maximum_spin_space


#The following is the scipy function that we tried first; it doesn't give out the correct answer
# Installing matlab proved to be more trouble than it's worth
# We had to resort to adapting the nullspace function from sympy instead

#from scipy.linalg import null_space

#print("null space:")
#nullspace = (null_space(max_spin_space.H))
#print(nullspace)

#np.concatenate(nullspace, max_spin_space)

import sympy

def nullspace(matrix):
  M = sympy.Matrix(matrix)
  K = M.nullspace()
  return -np.matrix(K).astype(complex)

#Format:
#A = np.matrix([[2.75,-1.2,0,3.2],[8.29,-4.8,7,0.01]])
#print(nullspace(A))


def get_total_spin_space(spins=spins):
  
  max = np.asmatrix(get_max_spin_space(spins=spins))

  nullsp = nullspace(max.H)

  while(np.count_nonzero(nullsp) !=0):
      normalized_vectors = []     
      vector = nullsp[0].H
      norm = np.linalg.norm(vector)
      normalized_vector = np.array(vector/norm)
      normalized_vectors.append(normalized_vector)

      while(np.count_nonzero(normalized_vector) != 0):
        vector = np.dot(comb_lowering_matrix("minus",spins=spins),normalized_vector)
        for i in range(len(vector)):
          if (abs(vector[i])<(10**-8)):
            vector[i] = 0
        norm = np.linalg.norm(vector)
        if (np.count_nonzero(vector) == 0):
          break
        normalized_vector = np.array(vector/norm)
        normalized_vectors.append(normalized_vector) 

      normalized_vectors = np.column_stack(normalized_vectors)
      
      max = np.hstack((max,normalized_vectors))

      nullsp = nullspace(max.H)


  total_spin_space = np.asmatrix(max).H
  return total_spin_space

# Returns the matrix of Clebsch-Gordan coefficients
def C_G_coefficients(spins=spins):
  max = get_total_spin_space(spins=spins).H
  signs = np.sign(max.real)
  squared = np.square(max)
  return np.multiply(squared,signs)



def state_column(given_spin, given_m, spins=spins):
  C_G = C_G_coefficients(spins=spins)
  total_spin = np.sum(spins)
  temp = total_spin
  place = 0
  index = 0
  while(temp != given_spin):
    place += 1
    temp -= 1
    if (temp < 0):
      raise Exception("Spin not present in columns")
  for i in range(place):
    index += (len(get_m(total_spin)))
  
  for i in (get_m(given_spin)):
    if given_m == i:
      break
    else:
      index += 1
  return C_G[:,index]


# Start of the main function

start = time.time()   # measuring the calculation time
coef = C_G_coefficients(spins)
end = time.time()
rcoef = coef.real     # removes the imaginary component 0 for cleaner display

print("spins:",spins)
print("C_G matrix:")
print(rcoef)

print("Calculation time:",end - start,"seconds")

def listCommands():
  print("List of available Commands:")
  print("help               : Display the list of available commands")
  print("matrix             : Display the Clebsch-Gordan coefficients matrix")
  print("spins              : Display the spins for each particle")
  print("restart            : Restart the Program with new particles  ")
  print("exit               : Exit the Program")

listCommands()

while(True):
  print("Please input your command")
  user_input = input(">")
  if user_input == "exit" or user_input == "quit":
    break
  elif user_input == "matrix":
    print("C_G matrix:")
    print(rcoef)
  elif user_input == "spins":
    print("spins: ",spins)
  elif user_input == "help":
    listCommands()
  elif user_input == "restart":
    spins = ask_for_spins()
    start = time.time()
    coef = C_G_coefficients(spins)
    end = time.time()
    rcoef = coef.real 
    print("spins:",spins)
    print("C_G matrix:")
    print(rcoef)
    print("Calculation time:",end - start,"seconds")

