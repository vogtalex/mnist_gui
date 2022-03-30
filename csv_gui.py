import csv
import os

csvName = "response.csv"
QAName = "QA.csv"

def writeToCSV(array):
    with open(csvName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(array)

def readFromCSV(array):
    with open(csvName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
        for row in reader:
            array.append(row)

def writeDataCSV(userPrediction, trueValue):
    array = [userPrediction, trueValue]
    writeToCSV(array)

def writeToCSV_QA(array):
    with open(QAName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow(array)

def outputCSV():
    outputArray = []
    readFromCSV(outputArray)
    for idx in range(len(outputArray)):
        print(outputArray[idx])

def initializeCSV():
    file = 'response.csv'
    if(os.path.exists(file) and os.path.isfile(file)):
        os.remove(file)

def formatCSV():
    csvName = 'formatted.csv'
    outputArray = []
    formattedArray = []
    numberArray = []
    totalNumString = ''
    correctNumString = ''
    correctCount = 0
    totalCount = 0
    readFromCSV(outputArray)

    for idx in range(len(outputArray)):
        if outputArray[idx][0] == outputArray[idx][1]:
            correctCount += 1
            correctNumString += str(outputArray[idx][1])
        totalNumString += str(outputArray[idx][1])
        totalCount += 1

    formattedArray.append(correctCount)
    formattedArray.append(totalCount)
    writeToCSV(formattedArray)

    for x in range(10):
        correct = correctNumString.count(str(x))
        total = totalNumString.count(str(x))
        numArray = [x, correct, total]
        writeToCSV(numArray)
