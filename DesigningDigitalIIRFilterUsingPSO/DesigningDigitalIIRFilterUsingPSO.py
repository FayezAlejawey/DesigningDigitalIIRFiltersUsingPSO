import math
import numpy as np
import pylab as plt

#####################Generic Functions#####################
def GetEulerFormula(omega, k):
    eulerFormula = np.add(np.cos(k*omega), np.array([complex(0, -i) for i in np.sin(k*omega)]))
    return eulerFormula

def GetRandomMatrix(numOfRows, numOfCols):
    return np.random.uniform(low=0.0, high=1.0, size=(numOfRows, numOfCols))

def GetLpNormApproximationError(theoriticalVal, actualVal, p):
    return np.power(np.sum(np.power(np.abs(theoriticalVal - np.abs(actualVal)), p)), 1/p)

def GetObjectiveFunction(*L_p_norm_args):
    objectiveFunction = L_p_norm_args[1]
    #for index in range(1, len(L_p_norm_args)):
        #objectiveFunction = np.add(objectiveFunction, L_p_norm_args[index])
    return objectiveFunction
###########################################################

#####################PSO Functions#########################
def UpdateVelocityMatrix(currentVelocityMatrix, pbestMatrix, gbestMatrix, positionMatrix, w):
    inertiaTerm = currentVelocityMatrix*w
    cognitiveTerm = 2*np.random.uniform(0, 1)*np.subtract(pbestMatrix, positionMatrix)
    socialTerm = 2*np.random.uniform(0, 1)*np.subtract(gbestMatrix, positionMatrix)
    return inertiaTerm + cognitiveTerm + socialTerm

def UpdatePositionMatrix(currentPositionMatrix, updatedVelocityMatrix):
    return np.add(currentPositionMatrix, updatedVelocityMatrix)

def CalculateFitnessFunctionValue(matrix, domainVal):
    #The most simplest fitness function, where it can be changed to any function wanted

    numOfRows = 0
    numOfCols = 0
    try:
        numOfRows = matrix.shape[0]
        numOfCols = matrix.shape[1]
    except:
        numOfRows = 1
        numOfCols = matrix.shape[0]

    if numOfRows == 1:
        theoriticalVal = GetTheoriticalValue(domainVal, matrix)
        L_1_norm = GetLpNormApproximationError(theoriticalVal, GetPiecewiseFunction(domainVal), 1)
        L_2_norm = GetLpNormApproximationError(theoriticalVal, GetPiecewiseFunction(domainVal), 2)
        objectiveFunction = GetObjectiveFunction(L_1_norm, L_2_norm)
        return np.sum(np.power(objectiveFunction, 2))

    fitnessArr = []
    for index in range(0, numOfRows):
        row = matrix[index, :]
        theoriticalVal = GetTheoriticalValue(domainVal, row)
        L_1_norm = GetLpNormApproximationError(theoriticalVal, GetPiecewiseFunction(domainVal), 1)
        L_2_norm = GetLpNormApproximationError(theoriticalVal, GetPiecewiseFunction(domainVal), 2)
        objectiveFunction = GetObjectiveFunction(L_1_norm, L_2_norm)
        fitnessArr.append(np.sum(np.power(objectiveFunction, 2)))

    return fitnessArr

def CalculategbestArray(pbest, domainVal):
    row = pbest[0, :]
    rowFitnessVal = CalculateFitnessFunctionValue(row, domainVal)
    gbestArr = row
    gbestVal = rowFitnessVal

    numOfRows = pbest.shape[0]
    for index in range(1, numOfRows):
        row = pbest[index, :]
        rowFitnessVal = CalculateFitnessFunctionValue(row, domainVal)
        if rowFitnessVal < gbestVal:
            gbestVal = rowFitnessVal
            gbestArr = row

    return gbestArr

def PSO(numOfRows, numOfCols, maxNumOfIterations, domainVal):
    #Initialization
    x = GetRandomMatrix(numOfRows, numOfCols)
    v = GetRandomMatrix(numOfRows, numOfCols)
    pbest = GetRandomMatrix(numOfRows, numOfCols)
    gbest = CalculategbestArray(pbest, domainVal)

    #Particles Updating
    slope = (0.9 - 0.4)/(1 - maxNumOfIterations)
    fitnessOfgbestArr = []
    arrToStopIterations = []
    for i in range(1, maxNumOfIterations + 1):
        #Calculating w where it has to be chosen such that
        #it start from 0.9 and it linearly decreases
        #until the end of the iterations where it will
        #become 0.4
        w = slope*(i - 1) + 0.9

        v = UpdateVelocityMatrix(v, pbest, np.tile(gbest, (numOfRows, 1)), x, w)
        x = UpdatePositionMatrix(x, v)

        fitnessOfxUpdated = CalculateFitnessFunctionValue(x, domainVal)
        fitnessOfpbest = CalculateFitnessFunctionValue(pbest, domainVal)
        for index in range(0, numOfRows):
            xRowFitnessVal = fitnessOfxUpdated[index]
            pbestRowFitnessVal = fitnessOfpbest[index]
            if xRowFitnessVal < pbestRowFitnessVal:
                pbest[index, :] = x[index, :]

        fitnessOfpbest = CalculateFitnessFunctionValue(pbest, domainVal)
        fitnessOfgbest = CalculateFitnessFunctionValue(gbest, domainVal)
        print(f"Iteration '{i}' out of '{maxNumOfIterations}' --> fitness(gbest) = {fitnessOfgbest}\n" + f"gbest = {gbest}\n\n")
        fitnessOfgbestArr.append(fitnessOfgbest)
        for index in range(0, numOfRows):
            pbestRowFitnessVal = fitnessOfpbest[index]
            if pbestRowFitnessVal < fitnessOfgbest:
                gbest = pbest[index, :]

        lengthOfFitnessgbestArr = len(fitnessOfgbestArr)
        if lengthOfFitnessgbestArr >= 2:
            gbest1 = fitnessOfgbestArr[lengthOfFitnessgbestArr - 1]
            gbest2 = fitnessOfgbestArr[lengthOfFitnessgbestArr - 2]
            if math.fabs(gbest1 - gbest2) < 0.0001:
                arrToStopIterations.append(1)
            else:
                arrToStopIterations.clear()

        if len(arrToStopIterations) > maxNumOfIterations/10:
            break

    return [gbest, fitnessOfgbestArr]
###########################################################

####################Filters Functions######################
def GetTheoriticalValue(domainVal, arr):
    return GetLpfTransferFunction(domainVal, arr)

def GetPiecewiseFunction(domain):
    return GetLpfPiecewiseFunction(domain)

def GetLpfPiecewiseFunction(domain):
    conds = [domain < 0, (domain >= 0) & (domain <= 0.2*math.pi), domain > 0.2*math.pi]
    funcs = [lambda domain: 0, lambda domain: 1, lambda domain: 0]
    return np.piecewise(domain, conds, funcs)

def GetLpfTransferFunction(omega, gbest):
    numer = np.abs(gbest[0] + gbest[1]*GetEulerFormula(omega, 1) + gbest[2]*GetEulerFormula(omega, 2) + gbest[3]*GetEulerFormula(omega, 3))
    denom = np.abs(1 + gbest[4]*GetEulerFormula(omega, 1) + gbest[5]*GetEulerFormula(omega, 2) + gbest[6]*GetEulerFormula(omega, 3))

    transferFun = np.divide(numer, denom)
    return transferFun
###########################################################

####################Digital IIR LPF########################
def DesignDigitalIirLpf():
    domainValue = np.linspace(0, math.pi, 100)
    normalizedDomainValue = domainValue/math.pi

    psoResult = PSO(50, 7, 10000, domainValue) 
    print(psoResult)

    gbest = psoResult[0]

    transferFun = GetTheoriticalValue(domainValue, gbest)
    normalizedTransferFunction = transferFun/np.max(transferFun)

    fig, axs = plt.plt.subplots(2)
    axs[0].plot(normalizedDomainValue, normalizedTransferFunction, color='red')
    axs[0].set_title('LP Filter Magnitude Response')

    iterations = list(range(0, len(psoResult[1])))
    axs[1].plot(iterations, psoResult[1])
    axs[1].set_title('Learning Curve')

    axisIndex = 0
    for ax in axs.flat:
        if axisIndex == 0:
            ax.set(xlabel='Normalized w', ylabel='|H(w)|')
        if axisIndex == 1:
            ax.set(xlabel='iterations', ylabel='Fitness(gbest)')
        axisIndex+=1

    plt.show()
###########################################################

#################Start of The Application##################
DesignDigitalIirLpf()
###########################################################