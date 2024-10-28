class bedpewriter():
    def __init__(self,file_path, resol):
        self.f = open(file_path,'w')
        self.resol = resol
    def write(self,chrom,x,y,prob,isbad=None):
        for i in range(len(x)):
            if isbad is not None and isbad[i]:
                # print('skip bad ',labels[i])
                pass
            else:
                if x[i] > y[i]:
                    pass
                else: 
                    self.f.write(chrom+'\t'+str(x[i])+'\t'+str(x[i]+self.resol)
                                +'\t'+chrom+'\t'+str(y[i])+'\t'+str(y[i]+self.resol)
                                +'\t'+str(prob[i])+'\n')