import csv
import os

csvName = "response.csv"

def writeToCSV(array):
    with open(csvName, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerows(array)

def initializeCSV():
    if(os.path.exists(csvName) and os.path.isfile(csvName)):
        os.remove(csvName)

def readFromCSV(array):
    with open(csvName, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', lineterminator='\n')
        for row in reader:
            array.append(row)

def outputCSV():
    outputArray = []
    readFromCSV(outputArray)
    for idx in range(len(outputArray)):
        print(outputArray[idx])
