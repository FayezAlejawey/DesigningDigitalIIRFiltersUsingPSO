import math
import numpy as np
import pylab as plt

#####################Generic Functions#####################
def GetEulerFormula(omega, k):
    eulerFormula = np.add(np.cos(k*omega), np.array([complex(0, -i) for i in np.sin(k*omega)]))
    return eulerFormula

def GetRandomMatrix(numOfRows, numOfCols):
    randMatrix = np.random.uniform(low=-1.0, high=1.0, size=(numOfRows, numOfCols))
    return randMatrix

def GetLpNormApproximationError(theoriticalVal, actualVal, p):
    return np.power(np.sum(np.power(np.abs(theoriticalVal - np.abs(actualVal)), p)), 1/p)

def GetRippleMagnitudes(actualVal, isPassBand):
    actualValMagnitude = np.abs(actualVal)
    if isPassBand:
        return np.max(actualValMagnitude) - np.min(actualValMagnitude)
    return np.max(actualValMagnitude)

def GetObjectiveFunction(*L_p_norm_args):
    objectiveFunction = L_p_norm_args[0]
    for index in range(1, len(L_p_norm_args)):
        objectiveFunction = np.add(objectiveFunction, L_p_norm_args[index])
    return objectiveFunction
###########################################################

#####################PSO Functions#########################
def UpdateVelocityMatrix(currentVelocityMatrix, pbestMatrix, gbestMatrix, positionMatrix, w):
    inertiaTerm = currentVelocityMatrix*w
    cognitiveTerm = 2*np.random.uniform(0, 1)*np.subtract(pbestMatrix, positionMatrix)
    socialTerm = 2*np.random.uniform(0, 1)*np.subtract(gbestMatrix, positionMatrix)
    return inertiaTerm + cognitiveTerm + socialTerm

def UpdatePositionMatrix(currentPositionMatrix, updatedVelocityMatrix):
    positionResults = np.add(currentPositionMatrix, updatedVelocityMatrix)
    return positionResults

def CalculateFitnessFunctionValue(matrix, domainVal, type):
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
        actualVal = GetActualFunction(domainVal, matrix, type)
        theoriticalFun = GetTheoriticalFunction(domainVal, matrix, type)
        L_1_norm = GetLpNormApproximationError(theoriticalFun, actualVal, 1)
        L_2_norm = GetLpNormApproximationError(theoriticalFun, actualVal, 2)
        rippleMagOfPassBand = GetRippleMagnitudes(actualVal, True)
        rippleMagOfStopBand = GetRippleMagnitudes(actualVal, False)
        objectiveFunction = GetObjectiveFunction(L_1_norm, L_2_norm, rippleMagOfPassBand, rippleMagOfStopBand)
        return objectiveFunction

    fitnessArr = []
    for index in range(0, numOfRows):
        row = matrix[index, :]
        actualVal = GetActualFunction(domainVal, row, type)
        theoriticalFun = GetTheoriticalFunction(domainVal, row, type)
        L_1_norm = GetLpNormApproximationError(theoriticalFun, actualVal, 1)
        L_2_norm = GetLpNormApproximationError(theoriticalFun, actualVal, 2)
        rippleMagOfPassBand = GetRippleMagnitudes(actualVal, True)
        rippleMagOfStopBand = GetRippleMagnitudes(actualVal, False)
        objectiveFunction = GetObjectiveFunction(L_1_norm, L_2_norm, rippleMagOfPassBand, rippleMagOfStopBand)
        fitnessArr.append(objectiveFunction)

    return fitnessArr

def CalculategbestArray(pbest, domainVal, type):
    row = pbest[0, :]
    rowFitnessVal = CalculateFitnessFunctionValue(row, domainVal, type)
    gbestArr = row
    gbestVal = rowFitnessVal

    numOfRows = pbest.shape[0]
    for index in range(1, numOfRows):
        row = pbest[index, :]
        rowFitnessVal = CalculateFitnessFunctionValue(row, domainVal, type)
        if rowFitnessVal < gbestVal:
            gbestVal = rowFitnessVal
            gbestArr = row

    return gbestArr

def PSO(numOfRows, numOfCols, maxNumOfIterations, domainVal, type):
    #Initialization
    x = GetRandomMatrix(numOfRows, numOfCols)
    v = GetRandomMatrix(numOfRows, numOfCols)
    pbest = GetRandomMatrix(numOfRows, numOfCols)
    gbest = CalculategbestArray(pbest, domainVal, type)

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

        fitnessOfxUpdated = CalculateFitnessFunctionValue(x, domainVal, type)
        fitnessOfpbest = CalculateFitnessFunctionValue(pbest, domainVal, type)
        for index in range(0, numOfRows):
            xRowFitnessVal = fitnessOfxUpdated[index]
            pbestRowFitnessVal = fitnessOfpbest[index]
            if xRowFitnessVal < pbestRowFitnessVal:
                pbest[index, :] = x[index, :]

        fitnessOfpbest = CalculateFitnessFunctionValue(pbest, domainVal, type)
        fitnessOfgbest = CalculateFitnessFunctionValue(gbest, domainVal, type)
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

        if len(arrToStopIterations) > maxNumOfIterations/5:
            break

    return [gbest, fitnessOfgbestArr]
###########################################################

####################Filters Functions######################
def GetActualFunction(domainVal, arr, type):
    if type == 'LR':
        return np.array([1.5, 3.7, 7.9, 15.2, 20.8, 25.3])
    if type == 'HPF':
        return GetHpfTransferFunction(domainVal, arr)
    if type == 'LPF':
        return GetLpfTransferFunction(domainVal, arr)

def GetTheoriticalFunction(domain, gbest, type):
    if type == 'LR':
        return GetLinearFunction(domain, gbest)
    if type == 'HPF':
        return GetHpfPiecewiseFunction(domain)
    if type == 'LPF':
        return GetLpfPiecewiseFunction(domain)

def GetLpfPiecewiseFunction(domain):
    conds = [domain < 0, (domain >= 0) & (domain <= 0.2*math.pi), domain > 0.2*math.pi]
    funcs = [lambda domain: 0, lambda domain: 1, lambda domain: 0]
    return np.piecewise(domain, conds, funcs)

def GetHpfPiecewiseFunction(domain):
    conds = [domain < 0.8*math.pi, (domain >= 0.8*math.pi) & (domain <= math.pi), domain > math.pi]
    funcs = [lambda domain: 0, lambda domain: 1, lambda domain: 0]
    return np.piecewise(domain, conds, funcs)

def GetLpfTransferFunction(omega, gbest):
    firstNumerator = 1 + gbest[0]*GetEulerFormula(omega, 1)
    firstDenominator = 1 + gbest[1]*GetEulerFormula(omega, 1)
    secondNumerator = 1 + gbest[2]*GetEulerFormula(omega, 1) + gbest[3]*GetEulerFormula(omega, 2)
    secondDenominator = 1 + gbest[4]*GetEulerFormula(omega, 1) + gbest[5]*GetEulerFormula(omega, 2)

    #numer = np.abs(gbest[0] + gbest[1]*GetEulerFormula(omega, 1) + gbest[2]*GetEulerFormula(omega, 2) + gbest[3]*GetEulerFormula(omega, 3))
    #denom = np.abs(1 + gbest[4]*GetEulerFormula(omega, 1) + gbest[5]*GetEulerFormula(omega, 2) + gbest[6]*GetEulerFormula(omega, 3))
    numerator = firstNumerator*secondNumerator
    denominator = firstDenominator*secondDenominator

    transferFun = np.divide(numerator, denominator)
    return transferFun

def GetHpfTransferFunction(omega, gbest):
    firstNumerator = 1 + gbest[0]*GetEulerFormula(omega, 1)
    firstDenominator = 1 + gbest[1]*GetEulerFormula(omega, 1)
    secondNumerator = 1 + gbest[2]*GetEulerFormula(omega, 1) + gbest[3]*GetEulerFormula(omega, 2)
    secondDenominator = 1 + gbest[4]*GetEulerFormula(omega, 1) + gbest[5]*GetEulerFormula(omega, 2)

    #numer = np.abs(gbest[0] + gbest[1]*GetEulerFormula(omega, 1) + gbest[2]*GetEulerFormula(omega, 2) + gbest[3]*GetEulerFormula(omega, 3))
    #denom = np.abs(1 + gbest[4]*GetEulerFormula(omega, 1) + gbest[5]*GetEulerFormula(omega, 2) + gbest[6]*GetEulerFormula(omega, 3))
    numerator = firstNumerator*secondNumerator
    denominator = firstDenominator*secondDenominator

    transferFun = np.divide(numerator, denominator)
    return transferFun

def GetLinearFunction(domain, gbest):
    y = gbest[0]*domain + gbest[1]
    return y
###########################################################

####################Digital IIR LPF########################
def DesignDigitalIirLpf(type):
    domainValue = np.linspace(0, math.pi, 100)
    normalizedDomainValue = domainValue/np.max(domainValue)

    psoResult = PSO(50, 6, 1000, domainValue, type) 
    print(psoResult)

    gbest = psoResult[0]

    transferFun = GetActualFunction(domainValue, gbest, type)
    magnitudeOfTransferFun = np.abs(transferFun)
    normalizedMagnitudeOfTransferFun = magnitudeOfTransferFun/np.max(magnitudeOfTransferFun)

    theoriticalLpf = GetTheoriticalFunction(domainValue, gbest, type)

    fig, axs = plt.plt.subplots(2)
    axs[0].plot(normalizedDomainValue, normalizedMagnitudeOfTransferFun, color='r')
    axs[0].plot(normalizedDomainValue, theoriticalLpf, color='b')
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

def DesignDigitalIirHpf(type):
    domainValue = np.linspace(0, math.pi, 100)
    normalizedDomainValue = domainValue/np.max(domainValue)

    psoResult = PSO(50, 6, 1000, domainValue, type) 
    print(psoResult)

    gbest = psoResult[0]

    transferFun = GetActualFunction(domainValue, gbest, type)
    magnitudeOfTransferFun = np.abs(transferFun)
    normalizedMagnitudeOfTransferFun = magnitudeOfTransferFun/np.max(magnitudeOfTransferFun)

    theoriticalLpf = GetTheoriticalFunction(domainValue, gbest, type)

    fig, axs = plt.plt.subplots(2)
    axs[0].plot(normalizedDomainValue, normalizedMagnitudeOfTransferFun, color='r')
    axs[0].plot(normalizedDomainValue, theoriticalLpf, color='b')
    axs[0].set_title('HP Filter Magnitude Response')

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

###################Linear Regression#######################
def ApproximateFirstOrderPolynomial(type):
    domainValue = np.array([0, 1, 2, 3, 4, 5])

    psoResult = PSO(50, 2, 1000, domainValue, type) 
    print(psoResult)

    gbest = psoResult[0]

    actualFun = GetActualFunction(domainValue, gbest, type)
    theoriticalFun = GetTheoriticalFunction(domainValue, gbest, type)

    fig, axs = plt.plt.subplots(2)
    axs[0].plot(domainValue, theoriticalFun, color='r')
    axs[0].scatter(domainValue, actualFun)
    axs[0].set_title('Approximate First Order Polynomial')

    iterations = list(range(0, len(psoResult[1])))
    axs[1].plot(iterations, psoResult[1])
    axs[1].set_title('Learning Curve')

    axisIndex = 0
    for ax in axs.flat:
        if axisIndex == 0:
            ax.set(xlabel='x', ylabel='y')
        if axisIndex == 1:
            ax.set(xlabel='iterations', ylabel='Fitness(gbest)')
        axisIndex+=1

    plt.show()
###########################################################

#############Function To Distinguish PSO Types#############
def CallPsoBasedOnType(type):
    if type == 'LR':
        ApproximateFirstOrderPolynomial(type)
    elif type == 'HPF':
        DesignDigitalIirHpf(type)
    elif type == 'LPF':
        DesignDigitalIirLpf(type)
###########################################################

#################Start of The Application##################
#For LPF Design Insert 'LPF'
#For HPF Design Insert 'HPF'
#For Linear Regression Insert 'LR'
CallPsoBasedOnType('LPF')
###########################################################