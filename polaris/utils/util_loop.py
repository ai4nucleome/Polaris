class bedpewriter():
    def __init__(self,file_path, resol, max_distance):
        self.f = open(file_path,'w')
        self.resol = resol
        self.max_distance = max_distance
    def write(self,chrom,x,y,prob):
        for i in range(len(x)):
            if x[i] < y[i] and y[i]-x[i] > 11*self.resol and y[i] - x[i] < self.max_distance:
                self.f.write(chrom+'\t'+str(x[i])+'\t'+str(x[i]+self.resol)
                            +'\t'+chrom+'\t'+str(y[i])+'\t'+str(y[i]+self.resol)
                            +'\t'+str(prob[i])+'\n')