from mrjob.job import MRJob
from mrjob.step import MRStep
from datetime import datetime
import sys

class MRAnom(MRJob):

    def steps(self):
        return [
          MRStep(mapper=self.mapperCoor, reducer=self.reducerCoor),
          MRStep(mapper=self.mapperAnom, reducer=self.reducerAnom)
        ]

    def mapperCoor(self, _, line):
        ID, Case_Number, Date, Block, IUCR, Primary_Type, Description, Location_Description, Arrest, Domestic, Beat, District, Ward, Community_Area, FBI_Code, X_Coordinate, Y_Coordinate, Year, Updated_On, Latitude, Longitude, Location = line.split(",", 21)


        if Longitude == "Longitude":
            Longitude = -87.6
        yield Longitude, 1

    def reducerCoor(self, key, values):
        yield key, sum(values)

    def mapperAnom(self, key, value):
        yield None, (key, value)

    def reducerAnom(self, key, values):
        sortedCounts = []
        for k, _ in values:
            if (k == None):
                continue
            try:
                sortedCounts.append(float(k))
            except:
                continue
                
        sortedCounts.sort(reverse=True)
        medCounts = int(len(sortedCounts)/2)
        medQ = int(medCounts/2)
        medianC = sortedCounts[medCounts]
        IQRN = sortedCounts[medCounts + medQ] - sortedCounts[medQ]
        
        
        cc = 0
        testScore = []
        for c in sortedCounts:
            test = (c - medianC)/IQRN
            if abs(test) >= 3:
                testScore.append((c, test))

        yield None, testScore
#         yield None, (cc, IQRN, sortedCounts[-1], medCounts, sortedCounts[medCounts], sortedCounts[medCounts + 1], np.std(sortedCounts), 

if __name__ == '__main__':
    start_time = datetime.now()  
    
    MRAnom.run()
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    output1 = "tid: " + str(elapsed_time)
    sys.stderr.write(output1)

