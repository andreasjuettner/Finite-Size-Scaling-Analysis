import numpy as np
import h5py
# define constants
Bbars = {'2':[0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59],'4':[0.42,0.43,0.44,0.45,0.46,0.47]}
Nlist     = [2,4]
glist     = [.1,.2,.3,.5,.6]
Llist     = [8,16,32,48,64,96,128]
# function for niceprint val(err)
def disperr3(val,dval):
    """ 
    based on Roland's script for MATLAB
    """
    n=len(val)
    if n!=len(dval):
     print("val and dval must have the same length!")
     print(val,dval)
     print("exiting")
     exit()
    dig=2
    res = n*['']
    for i in range(n):
     if dval[i] == 0. and val[i] == 0.:
      res[i]     = "0"
     elif np.isnan(val[i]) or np.isnan(dval[i]):
      res[i]     = "nan"
     elif dval[i] == 0. and val[i] != 0.:
      value      = "%d" % val[i]
      res[i]     = value
     #elif dval[i] < 10: 
     elif dval[i] < 1: 
      location 	 = int(np.floor(np.log10(dval[i])))
      append_err = "("+str(int(np.round(dval[i]*10**(-location+dig-1))))+")"
      #print location,val[i],dig
      if np.abs(val[i])<1e-100:
       val[i]=0.
       location=1
      valformat  = "%."+str(-location+dig-1)+"f"
      sval       = valformat % val[i]
      res[i]	 = sval +append_err
     elif dval[i]>=1:
      digits	 = min(0,int(np.ceil(np.log10(dval[i]))-1))+1
      error      = np.around(dval[i],digits)
      value 	 = np.around(val[i],digits)
      serr       = "%."+str(digits)+"f(%."+str(digits)+"f)"
      #print serr
      #print digits,dval[i],val[i],error,value,serr
      serr       = serr%(value,error)
      #print serr,dval[i]
      res[i]	 = serr#str(value)+"("+str(error)+")"
     else:
      digits	 = max(0,int(np.ceil(np.log10(dval[i]))-1))
      error 	 = int(round(dval[i]/10**digits)*10**digits)
      value 	 = round(val[i]/10**digits)*10**digits
      res[i]	 = str(value)+"("+str(error)+")"
    return res

# open data file
filename     = 'h5data/Bindercrossings.h5'
f = h5py.File(filename,'r')
tab = open('tables/crossings.tex','w')
tab.write('\\begin{tabular}{llllllllllllllllllllllllllllll}')
tab.write('\\hline\\hline')
for N in Nlist:
 tab.write('$L/a$&'+'&'.join([str(B) for B in Bbars[str(N)]])+'\\\\\n')
 tab.write('\\hline')
 for g in glist:
  for L in [8,16,32,48,64,96,128]:
   print("Loading data for N=%d, ag=%.2f, L/a=%d"%(N,g,L))
   s = []
   for Bbar in  Bbars[str(N)]:
    key = 'N=%d/g=%.2f/L=%d/Bbar=%.3f'%(N,g,L,Bbar) 
    try:
     f[key+'/central']
    except:
     continue
    s.append(disperr3([f[key+'/central'].value],[f[key+'/std'].value])[0])
   tab.write(str(L)+'&'+'&'.join(s)+'\\\\\n')
  tab.write('\\hline')
 tab.write('\\hline')
tab.write('\\end{tabular}')
tab.close()
