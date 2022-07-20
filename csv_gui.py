import csv
import os

csvName = "response.csv"
QAName = "QA.csv"

def writeToCSV(array):
    with open(csvName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(array)

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
    file = csvName
    if(os.path.exists(file) and os.path.isfile(file)):
        os.remove(file)

def formatCSV(userInput):
    header = ['UserInput']
    data = [userInput]

    with open('countries.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(data)
