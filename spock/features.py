from collections import OrderedDict
from spock.ClassifierSeries import getsecT
import numpy as np
import math
import rebound


class Trio:
    def __init__(self, trio, sim):
        '''initializes new set of features.
        
            note: each list of the key is the series of data points, 
                  second dict is for final features
        '''
        # We keep track of the this trio and the adjacent pairs
        self.trio = trio
        self.pairs = get_pairs(sim, trio)
        
        # innitialize running list which keeps track of data during simulation
        self.runningList = OrderedDict()
        self.runningList['time'] = []
        self.runningList['MEGNO'] = []
        self.runningList['threeBRfill'] = []



        for each in ['near','far']:
            self.runningList['EM' + each] = []
            self.runningList['EP' + each] = []
            self.runningList['MMRstrength' + each] = []


        

    #returned features
        self.features = OrderedDict()

        for each in ['near', 'far']:
            self.features['EMcross' + each] = np.nan
            self.features['EMfracstd' + each] = np.nan
            self.features['EPstd' + each] = np.nan
            self.features['MMRstrength' + each] = np.nan

        self.features['threeBRfillfac'] = np.nan
        self.features['threeBRfillstd'] = np.nan
        self.features['MEGNO'] = np.nan
        self.features['MEGNOstd'] = np.nan
        self.features['ThetaSTD12'] = np.nan
        self.features['ThetaSTD23'] = np.nan
        self.features['ThetaSTD12alt'] = np.nan
        self.features['ThetaSTD23alt'] = np.nan
        self.features['Tsec'] = np.nan
        self.features['near'] = np.nan
        self.features['nearThetaSTD'] = np.nan
        self.features['nearThetaSTDalt'] = np.nan


        

    def fillVal(self, Nout):
        '''Fills with nan values
        
            Arguments: 
                Nout: number of datasets collected
        '''
        for each in self.runningList.keys():
            self.runningList[each] = [np.nan] * Nout
        self.theta12 = np.zeros(Nout)
        self.theta23 = np.zeros(Nout)
        self.theta12alt = np.zeros(Nout)
        self.theta23alt = np.zeros(Nout)
        self.p2p1 = np.zeros(Nout)
        self.p3p2 = np.zeros(Nout)
        self.e1 = np.zeros(Nout)
        self.e2 = np.zeros(Nout)
        self.e3 = np.zeros(Nout)
        self.l1 = np.zeros(Nout)
        self.l2 = np.zeros(Nout)
        self.l3 = np.zeros(Nout)
        self.pomegarel12 = np.zeros(Nout)
        self.pomegarel23 = np.zeros(Nout)

    def getNum(self):
        '''Returns number of features collected as ran'''
        return len(self.runningList.keys())

    def populateData(self, sim, minP,i):
        '''Populates the runningList data dictionary for one time step.
        
            Note: must specify how each feature is calculated and added
        '''
        ps = sim.particles
        
        for q, [label, i1, i2] in enumerate(self.pairs):
            m1 = ps[i1].m
            m2 = ps[i2].m
            #calculate eccentricity vector
            e1x, e1y = ps[i1].e * np.cos(ps[i1].pomega), ps[i1].e * np.sin(ps[i1].pomega)
            e2x, e2y = ps[i2].e * np.cos(ps[i2].pomega), ps[i2].e * np.sin(ps[i2].pomega)
            
            self.runningList['time'][i]= sim.t/minP
            #crossing eccentricity
            self.runningList['EM'+label][i] = np.sqrt((e2x - e1x)**2 + (e2y - e1y)**2)
            #mass weighted crossing eccentricity
            self.runningList['EP'+label][i] = np.sqrt((m1 * e1x + m2 * e2x)**2 + 
                                                      (m1 * e1y + m2 * e2y)**2) / (m1+m2)
            #calculate the strength of MMRs
            MMRs = find_strongest_MMR(sim, i1, i2)
            self.runningList['MMRstrength' + label][i] = MMRs[2]
        
        self.p2p1[i] = ((ps[2].P/ps[1].P))
        self.p3p2[i]=((ps[3].P/ps[2].P))
        self.e1[i]=(ps[1].e)
        self.e2[i]=(ps[2].e)
        self.e3[i]=(ps[3].e)
        self.l1[i]=(ps[1].l)
        self.l2[i]=(ps[2].l)
        self.l3[i]=(ps[3].l)
        self.pomegarel12[i]=(getPomega(sim,1,2))
        self.pomegarel23[i]=(getPomega(sim,2,3))
        self.runningList['threeBRfill'][i]= threeBRFillFac(sim, self.trio)
            
        
        # check rebound version, if old use .calculate_megno, otherwise use .megno, old is just version less then 4
        if float(rebound.__version__[0]) < 4:
            self.runningList['MEGNO'][i] = sim.calculate_megno()
        else:
            self.runningList['MEGNO'][i] = sim.megno()

        



    def startingFeatures(self, sim):
        '''Initializes, adding to the features that only depend on initial conditions'''

        ps = sim.particles
        for [label, i1, i2] in self.pairs:  
            # calculate crossing eccentricity
            self.features['EMcross' + label] = (ps[i2].a - ps[i1].a) / ps[i1].a
        # calculate secular timescale and adds feature
        self.features['Tsec']= getsecT(sim, self.trio)

    def fill_features(self, args):
        '''fills the final set of features that are returned to the ML model.
            
            Each feature is filled depending on some combination of runningList features and initial condition features
        '''
        Norbits = args[0]
        Nout = args[1]
        trio = args[2]

        # if system becomes unstable, finds up until what time we have data for
        # we will only fill using this data
        end = (self.runningList['time'] + [np.nan]).index(np.nan)

        #if not np.isnan(self.runningList['MEGNO']).any(): # no nans
        self.features['MEGNO']= np.median(self.runningList['MEGNO'][int(0.9 * end):end]) # smooth last 10% to remove oscillations around 2
        self.features['MEGNOstd']= np.std(self.runningList['MEGNO'][int(end/5):end])

        for label in ['near', 'far']: 
            # cut out first value (init cond) to avoid cases
            # where user sets exactly b*n2 - a*n1 & strength is inf
            self.features['MMRstrength'+label] = np.median(self.runningList['MMRstrength'+label][:end])
            self.features['EMfracstd'+label] = np.std(self.runningList['EM'+label][:end])/ self.features['EMcross'+label]
            self.features['EPstd'+label] = np.std(self.runningList['EP'+label][:end])
        ############
        Pratio12 = 1/np.median(self.p2p1[np.nonzero(self.p2p1)])
        Pratio32 = 1/np.median(self.p3p2[np.nonzero(self.p3p2)])
        pval12 = getval(Pratio12)
        pval32 = getval(Pratio32)
        #print(pomegarel12)
        for x in range(Nout):
            self.theta12[x]=calcTheta(self.l1[x],self.l2[x],self.pomegarel12[x],pval12)
            self.theta23[x]=calcTheta(self.l2[x],self.l3[x],self.pomegarel23[x],pval32)
            self.theta12alt[x]=calcThetaALT(self.l1[x],self.l2[x],self.pomegarel12[x],pval12)
            self.theta23alt[x]=calcThetaALT(self.l2[x],self.l3[x],self.pomegarel23[x],pval32)
        self.features['ThetaSTD12']= np.std(self.theta12)
        self.features['ThetaSTD23']= np.std(self.theta23)
        self.features['ThetaSTD12alt']= np.std(self.theta12alt)
        self.features['ThetaSTD23alt']= np.std(self.theta23alt)
        self.features['near'] = self.pairs[0][1:]
        if trio[0] == self.pairs[0][1]:
            self.features['nearThetaSTD'] = self.features['ThetaSTD12']
            self.features['nearThetaSTDalt']= self.features['ThetaSTD12alt']
        else:
            self.features['nearThetaSTD'] = self.features['ThetaSTD23']
            self.features['nearThetaSTDalt']= self.features['ThetaSTD23alt']

        
        self.features['threeBRfillfac']= np.mean(self.runningList['threeBRfill'])
        self.features['threeBRfillstd']= np.std(self.runningList['threeBRfill'])
    

            


 ######################### Taken from celmech github.com/shadden/celmech
def farey_sequence(n):
    """Return the nth Farey sequence as order pairs of the form (N,D) where `N' is the numerator and `D' is the denominator."""
    a, b, c, d = 0, 1, 1, n
    sequence=[(a,b)]
    while (c <= n):
        k = int((n + b) / d)
        a, b, c, d = c, d, (k*c-a), (k*d-b)
        sequence.append( (a,b) )
    return sequence
def resonant_period_ratios(min_per_ratio,max_per_ratio,order):
    """Return the period ratios of all resonances up to order 'order' between 'min_per_ratio' and 'max_per_ratio' """
    if min_per_ratio < 0.:
        raise AttributeError("min_per_ratio of {0} passed to resonant_period_ratios can't be < 0".format(min_per_ratio))
    if max_per_ratio >= 1.:
        raise AttributeError("max_per_ratio of {0} passed to resonant_period_ratios can't be >= 1".format(max_per_ratio))
    minJ = int(np.floor(1. / (1. - min_per_ratio)))
    maxJ = int(np.ceil(1. / (1. - max_per_ratio)))
    res_ratios=[(minJ-1,minJ)]
    for j in range(minJ,maxJ):
        res_ratios = res_ratios + [ ( x[1] * j - x[1] + x[0] , x[1] * j + x[0]) for x in farey_sequence(order)[1:] ]
    res_ratios = np.array(res_ratios)
    msk = np.array( list(map( lambda x: min_per_ratio < x[0] / float(x[1]) < max_per_ratio , res_ratios )) )
    return res_ratios[msk]
##########################

# sorts out which pair of planets has a smaller EMcross, labels that pair inner, other adjacent pair outer
# returns a list of two lists, with [label (near or far), i1, i2], where i1 and i2 are the indices, with i1 
# having the smaller semimajor axis

def get_pairs(sim, trio):
    ''' returns the three pairs of the given trio.
    
    Arguments:
        sim: simulation in question
        trio: indicies of the 3 particles in question, formatted as [p1, p2, p3]
    return: returns the two pairs in question, formatted as 
                [[near pair, index, index], [far pair, index, index]]
    '''
 
    ps = sim.particles
    sortedindices = sorted(trio, key = lambda i: ps[i].a) # sort from inner to outer
    EMcrossInner = ((ps[sortedindices[1]].a - ps[sortedindices[0]].a)
                    / ps[sortedindices[0]].a)

    EMcrossOuter = ((ps[sortedindices[2]].a - ps[sortedindices[1]].a)
                    / ps[sortedindices[1]].a)

    if EMcrossInner < EMcrossOuter:
        return [['near', sortedindices[0], sortedindices[1]],
                ['far', sortedindices[1], sortedindices[2]]]
    else:
        return [['near', sortedindices[1], sortedindices[2]],
                ['far', sortedindices[0], sortedindices[1]]]

# taken from original spock, some comments changed
####################################################
def find_strongest_MMR(sim, i1, i2):
    '''Finds the strongest MMR between two planets

        Arguments:
            sim: the simulation in question
            i1: the inner most of the two planets in question
            i2: the outer most of the two planets in question
        return: information about the resonance, the third item (index 2)
                is the maximum strength of the resonance between planets
    '''
    maxorder = 2
    ps = sim.particles
    n1 = ps[i1].n
    n2 = ps[i2].n

    m1 = ps[i1].m / ps[0].m
    m2 = ps[i2].m / ps[0].m

    Pratio = n2 / n1

    delta = 0.03
    if Pratio < 0 or Pratio > 1: # n < 0 = hyperbolic orbit, Pratio > 1 = orbits are crossing
        return np.nan, np.nan, np.nan

    minperiodratio = max(Pratio - delta, 0.)
    maxperiodratio = min(Pratio + delta, 0.99) # too many resonances close to 1
    res = resonant_period_ratios(minperiodratio, maxperiodratio, order=maxorder)

    # Calculating EM exactly would have to be done in celmech for each j/k res below, and would slow things down. This is good enough for approx expression
    EM = np.sqrt((ps[i1].e * np.cos(ps[i1].pomega) - ps[i2].e * np.cos(ps[i2].pomega))**2 + 
                 (ps[i1].e * np.sin(ps[i1].pomega) - ps[i2].e * np.sin(ps[i2].pomega))**2)
    
    EMcross = (ps[i2].a - ps[i1].a) / ps[i1].a

    j, k, maxstrength = np.nan, np.nan, 0 
    for a, b in res:
        nres = (b * n2 - a * n1) / n1
        if nres == 0:
            s = np.inf # still want to identify as strongest MMR if initial condition is exatly b*n2-a*n1 = 0
        else:
            s = np.abs(np.sqrt(m1 + m2) * (EM / EMcross)**((b - a) / 2.) / nres)
        if s > maxstrength:
            j = b
            k = b-a
            maxstrength = s
    if maxstrength == 0:
        maxstrength = np.nan

    return j, k, maxstrength
##############################################

def swap(a, b):
    '''Simple swap function'''
    return b, a
def calcTheta(la,lb,pomegarel, val):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    return np.mod(theta, 2*np.pi)
def calcThetaALT(la,lb,pomegarel, val):
    theta = (val[1]*lb) -(val[0]*la)-(val[1]-val[0])*pomegarel
    # maybe to check for ocilations around 0 since will wrap around
    return np.mod(np.mod(theta, 2*np.pi)-np.pi,2*np.pi)

#WARNING VERY SLOW
#import fractions
def getval( Pratio: list):
    maxorder = 5
    delta = 0.03
    minperiodratio = Pratio-delta
    maxperiodratio = Pratio+delta # too many resonances close to 1
    if maxperiodratio >.999:
        maxperiodratio =.999
    res = resonant_period_ratios(minperiodratio,maxperiodratio, order=maxorder)
    val = [10000000,10]
    for i,each in enumerate(res):
        if np.abs((each[0]/each[1])-Pratio)<np.abs((val[0]/val[1])-Pratio):
            #which = i
            
            val = each
    
    # frac = fractions.Fraction(Pratio).limit_denominator(40)
    # val = frac.numerator, frac.denominator

    return val



def getPomega(sim, i1, i2):
    ps = sim.particles
    evec2 = ps[i2].e*np.exp(1j*ps[i2].pomega)
    evec1 = ps[i1].e*np.exp(1j*ps[i1].pomega)
    erel = evec2-evec1
    pomegarel=np.angle(erel)
    return pomegarel


def imagMaxErel(sim, trio):
    ecom, e13, emin, [chi12t, chi23t] = getEigMode(sim, trio)

    v13 = complex(e13[0],e13[1])
    vm = complex(emin[0],emin[1])
    v12 = chi23t*v13-vm
    v23 = chi12t*v13+vm
    return v12,v23,v13

def getEigMode(sim, trio):
    '''returns the three eigen modes for a three body system when given trion.
        
        return: ecom, e13, emin, [chi12t,chi23t]
    '''
    #FIXME
    [i1,i2,i3] = trio
    ps = sim.particles
    p1, p2, p3 = ps[i1], ps[i2], ps[i3]

    ecom, e13, emin = [np.nan, np.nan],[np.nan, np.nan],[np.nan, np.nan]

    m1, m2, m3 = p1.m, p2.m, p3.m
    m_tot = m1 + m2 + m3
    mu1, mu2, mu3 = m1/m_tot, m2/m_tot, m3/m_tot
    
    #alpha is semi major axis ratio
    alpha12, alpha23 = p1.a/p2.a, p2.a/p3.a
    alpha13 = alpha12*alpha23
    #crossing ecc in Appendix A
    ec12 = alpha12**(-1/4)*alpha23**(3/4)*alpha23**(-1/8)*(1-alpha12)
    ec23 = alpha23**(-1/2)*alpha12**(1/8)*(1-alpha23)
    ec13 = alpha13**(-1/2)*(1-alpha13)
    #made up constant
    eta = (ec12 - ec23)/ec13
    chi12 = mu1*(1-eta)**3*(3+eta)
    chi23 = mu3*(1+eta)**3*(3-eta)
    chi12t = chi12/(chi12+chi23)
    chi23t = chi23/(chi12+chi23)

    
    #rel ecc vec
    e1x, e2x, e3x = [p.e*np.cos(p.pomega) for p in [p1,p2,p3]]
    e1y, e2y, e3y = [p.e*np.sin(p.pomega) for p in [p1,p2,p3]]

    emin = np.array([chi23t*(e3x-e2x)-chi12t*(e2x-e1x), chi23t*(e3y-e2y)-chi12t*(e2y-e1y)])
    e13 = np.array([e3x-e1x, e3y-e1y])
    ecom = np.array([(mu1*e1x+mu2*e2x+mu3*e3x), (mu1*e1y+mu2*e2y+mu3*e3y)])

    return ecom, e13, emin, [chi12t,chi23t]

def getZcrit(sim, i1, i2):
    ps = sim.particles
    p1 = ps[i1]
    p2 = ps[i2]
    t1 = ((p2.a-p1.a)/p2.a)/math.sqrt(2)
    m1 = p1.m/ps[0].m
    m2 = p2.m/ps[0].m
    exp = -2.2*((m1+m2)**(1/3))*((p2.a/(p2.a-p1.a))**(4/3))


def threeBRFillFac(sim, trio):
    '''calculates the 3BR filling factor in acordance to petit20'''
    ps = sim.particles
    b0, b1,b2,b3 = ps[0], ps[trio[0]], ps[trio[1]], ps[trio[2]]
    m0,m1,m2,m3 = b0.m,b1.m,b2.m,b3.m
    ptot = None

    #semim
    a12 =(b1.a/b2.a)
    a23 = (b2.a/b3.a)

    #equation 43
    d12 = 1- a12
    d23 = 1- a23

    #equation 45
    d = (d12*d23)/(d12+d23)

    #equation 19
    mu12 = b1.P/b2.P
    mu23 = b2.P/b3.P

    #equation 21
    eta = (mu12*(1-mu23))/(1-(mu12*mu23))

    #equation 53
    eMpow2 = (m1*m3 + m2*m3*(eta**2)*(a12**(-2))+m1*m2*(a23**2)*((1-eta)**2))/(m0**2)

    #equation 59
    dov = ((42.9025)*(eMpow2)*(eta*((1-eta)**3)))**(0.125)

    #equation 60

    ptot = (dov/d)**4

    return abs(ptot)