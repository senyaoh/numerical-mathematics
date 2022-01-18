import numpy as np
import matplotlib.pyplot as plt

#Aufgabe 4.1
def newton(F, dF, x0, delta=10**-4, epsilon=10**-4, maxIter=100):
    xk = x0
    for i in range(maxIter):

        Fxk = F(xk) # I swear I'm not trying to spell a certain curse word
        dFxk = dF(xk)

        if len(x0) == 1: # Can't use linalg.solve if it's 1d
            deltaxk = np.array([-Fxk/dFxk])
        else:
            deltaxk = np.linalg.solve(dFxk, -Fxk)
        xkplus1 = xk + deltaxk

        if np.linalg.norm(xkplus1 - xk) < delta:
            # print("Die Iteration stagniert, das Newton-Verfahren wird abgebrochen.")
            return xkplus1

        if np.linalg.norm(F(xkplus1)) < epsilon:
            # print("Die Funktionswerte verschwinden, das Newton-Verfahren wird abgebrochen.")
            return xkplus1

        xk = xkplus1

    # print("Die maximale Anzahl an Iterationen ist erreicht, das Newton-Verfahren wird abgebrochen.")
    return xk

#Aufgabe 4.2
def F2(x: np.array) -> np.array:
    f = lambda x: x**3 - 2*x
    Fx = f(x[0])
    return np.array(Fx)

def dF2(x: np.array) -> np.array:
    f = lambda x: 3*x**2 - 2
    Fx = f(x[0])
    return np.array(Fx)

#Aufgabe 4.3
def F3(x: np.array) -> np.array:
    f1 = lambda x1, x2: x1**2 + x2**2 - 6*x1
    f2 = lambda x1, x2: 3/4 * np.exp(-x1) - x2
    F = [f1, f2]
    Fx = [f(x[0], x[1]) for f in F]
    return np.array(Fx).T

def dF3(x: np.array) -> np.array:
    (x1, x2) = x
    return np.array([[2*x1 - 6, 2*x2], [-3/4*np.exp(-x1), -1]])

#Aufgabe 4.4
def discrete_b_set(x_left=-1, x_right=1, y_top=1, y_bottom=-1, x_size=512, y_size=512):
    x = np.linspace(x_left, x_right, x_size)
    y = np.linspace(y_top, y_bottom, y_size)
    xv, yv = np.meshgrid(x, y)
    B = xv + 1j*yv
    return B

def F4(x: np.array) -> np.array:
    f = lambda x: x**3 - 1
    Fx = f(x[0])
    return np.array(Fx)

def dF4(x: np.array) -> np.array:
    f = lambda x: 3*x**2
    Fx = f(x[0])
    return np.array(Fx)

def color_code_root(x):
    r1 = np.absolute(x - 1)
    r2 = np.absolute(x - (-1/2 + 0.866j))
    r3 = np.absolute(x - (1/2 - 0.866j))
    data = [((53,42,135), r1), ((49,176,154), r2), ((249,251,14), r3)]
    data.sort(key=lambda x:x[1])
    return data[0]

#Aufgabe 4.5
def F5(x: np.array) -> np.array:
    f = lambda x: x**5 - 1
    Fx = f(x[0])
    return np.array(Fx)

def dF5(x: np.array) -> np.array:
    f = lambda x: 5*x**4
    Fx = f(x[0])
    return np.array(Fx)

#Aufgabe 4.6
def divF6(x: np.array) -> np.array:
    f1 = lambda x1, x2: 4*(x1+1)**3
    f2 = lambda x1, x2: 4*(x2-1)**3
    F = [f1, f2]
    Fx = [f(x[0], x[1]) for f in F]
    return np.array(Fx).T

def HF6(x: np.array) -> np.array:
    (x1, x2) = x
    return np.array([[12*(x1+1)**2, 0], [0, 12*(x2-1)**2]])

#Aufgabe 4.7
def mandelbrot(c, max_iter):
    z = c
    i = 0
    
    while np.absolute(z)<=2 and i < max_iter:
        z = z**2 + c
        i += 1
    return i

if __name__ == "__main__":
    # Aufgabe 4.2: Solve R -> R equation
    print("Aufgabe 4.2")
    print(newton(F2, dF2, np.array([1]), 10**-10, 10**-10, 50))
    print(newton(F2, dF2, np.array([0]), 10**-10, 10**-10, 50))
    print(newton(F2, dF2, np.array([2]), 10**-10, 10**-10, 50))
    print(newton(F2, dF2, np.array([-2]), 10**-10, 10**-10, 50))

    # Aufgabe 4.3: Solve R^2 -> R^2 equations
    print("Aufgabe 4.3")
    print(newton(F3, dF3, np.array([0.08, 0.7])))

    # Aufgabe 4.4
    print("Aufgabe 4.4")
    B = discrete_b_set()
    Bf = B.flatten()
    F4result = [np.around(newton(F4, dF4, np.array([x]), 10**-5, 10**-5, 15)[0],2) for x in Bf]
    colormap = [color_code_root(x)[0] for x in F4result]
    F4img = np.reshape(colormap, (512, 512, 3))
    plt.figure(figsize = (20,8))
    plt.imshow(F4img)
    plt.show()

    # Aufgabe 4.5
    print("Aufgabe 4.5")
    F5result = [newton(F5, dF5, np.array([x]), 10**-14, 10**-14, 5)[0] for x in Bf]
    F5phase = [np.angle(z) for z in F5result]
    F5phase_min_abs = abs(min(F5phase))
    F5phase_max = max(F5phase)
    F5mapcolor = [int((F5phase_min_abs + x) / (F5phase_min_abs + F5phase_max) * 255) for x in F5phase]
    F5img = np.reshape(F5mapcolor, (512, 512))
    plt.figure(figsize = (20,8))
    plt.imshow(F5img, cmap='jet', interpolation='nearest')
    plt.show()


    # Aufgabe 4.6
    print("Aufgabe 4.6")
    print(newton(divF6, HF6, np.array([-1.1, 1.1])))

    # Aufgabe 4.7
    print("Aufgabe 4.7")
    MB = discrete_b_set(-1.5, 0.5, 1, -1, 1024, 1024)
    MBf = MB.flatten()
    Mresult = [mandelbrot(c, 256) for c in MBf]
    Mimg = np.reshape(Mresult, (1024, 1024))
    plt.figure(figsize = (20,8))
    plt.imshow(Mimg)
    plt.show()

    