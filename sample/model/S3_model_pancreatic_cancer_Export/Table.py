class CubicSpline:
    "Cubic Spline class"

    def __init__(self,knots,knotHeights,numSplines,splineCoeffs,boundaryCondition):
        self.knots=knots
        self.knotHeights=knotHeights
        self.numSplines=numSplines
        self.splineCoeffs=splineCoeffs
        self.boundaryCondition=boundaryCondition

    def evaluate(self,x):
        y=float('nan')
        #Find domain
        index=-1
        if(x<self.knots[0]): #Extrapolate left
            x=x-self.knots[0]
            a=self.splineCoeffs[0]
            if(self.boundaryCondition==0 or self.boundaryCondition==1): #Natural or clamped
                slope=a[0,1]
                y=slope*x+self.knotHeights[0]
            else: #Not-a-knot or periodic
                index=0
                y=self.splineCoeffs[index,0]+self.splineCoeffs[index,1]*x+self.splineCoeffs[index,2]*x*x+self.splineCoeffs[index,3]*x*x*x
        elif(x>self.knots[self.numSplines]): #Extrapolate right
            a=self.splineCoeffs[self.numSplines-1]
            if(self.boundaryCondition==0 or self.boundaryCondition==1): #Natural or clamped
                x=x-self.knots[sel.numSplines]
                h=self.knots[self.numSplines]-self.knots[self.numSplines-1]
                slope=a[0,1]+2*a[0,2]*h+3*a[0,3]*h*h
                y=slope*x+self.knotHeights[self.numSplines]
            else: #Not-a-knot or periodic
                index=self.numSplines-1
                x=x-self.knots[index]
                y=self.splineCoeffs[index,0]+self.splineCoeffs[index,1]*x+self.splineCoeffs[index,2]*x*x+self.splineCoeffs[index,3]*x*x*x
        else: #Interpolate
            index=0
            while(x>self.knots[index+1] and index<self.numSplines-1):
                index+=1
            x=x-self.knots[index]
            y=self.splineCoeffs[index,0]+self.splineCoeffs[index,1]*x+self.splineCoeffs[index,2]*x*x+self.splineCoeffs[index,3]*x*x*x
        return(y)

class Table:
    "Table class"
    name=""
    type=""
    lookupMethod=""
    interpolate=""
    boundary=""
    extrapolate=""
    numRows=0
    numCols=0

    def __init__(self,name,type,lookupMethod,interpolate,boundary,extrapolate,numRows,numCols,filepath):
        self.name=name
        self.type=type
        self.lookupMethod=lookupMethod
        self.interpolate=interpolate
        self.boundary=boundary
        self.extrapolate=extrapolate
        self.numRows=numRows
        self.numCols=numCols
        self.splines=[]
        #Read table
        with open(filepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            self.headers = next(reader)  #Headers
            self.data=[]
            for r in range(0,numRows):
                strRow=next(reader)
                curRow=[float(x) for x in strRow]
                self.data.append(curRow)

    def getLookupValue(self,index,col):
        if(col<1 or col>(self.numCols-1)): #Invalid column
            return(float('nan'))
        else: #Valid column
            val=float('nan')
            if(self.lookupMethod=="Exact"):
                row=-1
                found=False
                while(found==False and row<self.numRows):
                    row+=1
                    if(index==self.data[row][0]): found=True
                if(found): val=self.data[row][col]
            elif(self.lookupMethod=="Truncate"):
                if(index<self.data[0][0]): val=float('nan') #Below first value - error
                elif(index>=self.data[self.numRows-1][0]): val=self.data[self.numRows-1][col] #Above last value
                else: #Between
                    row=0
                    while(self.data[row][0]<index): row+=1
                    if(index==self.data[row][0]): val=self.data[row][col]
                    else: val=self.data[row-1][col]
            elif(self.lookupMethod=="Interpolate"):
                if(self.interpolate=="Linear"):
                    if(index<=self.data[0][0]): #Below or at first index
                        slope=(self.data[1][col]-self.data[0][col])/(self.data[1][0]-self.data[0][0])
                        val=self.data[0][col]-(self.data[0][0]-index)*slope
                    elif(index>self.data[self.numRows-1][0]): #Above last index
                        slope=(self.data[self.numRows-1][col]-self.data[self.numRows-2][col])/(self.data[self.numRows-1][0]-self.data[self.numRows-2][0])
                        val=self.data[self.numRows-1][col]+(index-self.data[self.numRows-1][0])*slope
                    else: #Between
                        row=0
                        while(self.data[row][0]<index):row+=1
                        slope=(self.data[row][col]-self.data[row-1][col])/(self.data[row][0]-self.data[row-1][0])
                        val=self.data[row-1][col]+(index-self.data[row-1][0])*slope
                elif(self.interpolate=="Cubic Splines"):
                    val=self.splines[col-1].evaluate(index)
                
                #Check extrapolation conditions
                if(self.extrapolate=="No"):
                    if(index<=self.data[0][0]): val=self.data[0][col] #Below or at first index
                    elif(index>self.data[self.numRows-1][0]): val=self.data[self.numRows-1][col] #Above last index
                elif(self.extrapolate=="Left only"): #truncate right
                    if(index>self.data[self.numRows-1][0]): val=self.data[self.numRows-1][col] #Above last index
                elif(self.extrapolate=="Right only"): #truncate left
                    if(index<=self.data[0][0]): val=self.data[0][col] #Below or at first index
            return(val)

    def calcEV(self,col):
        ev=0
        for r in range(0,self.numRows):
            ev+=self.data[r][0]*self.data[r][col]
        return(ev)

